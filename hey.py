"""
Vehicle Counting per Lane with YOLO + SORT
------------------------------------------
Run:
    python hey.py

What it does:
- Loads YOLOv8 COCO model to detect vehicles
- Tracks with SORT to avoid double counting
- Defines 3 lanes (split equally) and counts vehicles per lane
- Saves CSV: VehicleID, Lane, Frame, Timestamp
- Annotated video with lane overlays + final summary
"""

import os
import csv
import time
import math
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# -------------------- SORT Tracker --------------------
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
    return o

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in range(len(y)) if y[i] >= 0])
    except Exception:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / (h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = math.sqrt(max(0.0, x[2].item() * x[3].item()))
    h = max(1e-6, w / (x[3].item() + 1e-6))
    x1 = x[0] - w/2.
    y1 = x[1] - h/2.
    x2 = x[0] + w/2.
    y2 = x[1] + h/2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]], dtype=float)
        self.kf.R[2:,2:] *= 10.0
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:4])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# -------------------- Lane Drawing --------------------
def lane_index_from_point(x, w):
    if x < w/3:
        return 1
    elif x < 2*w/3:
        return 2
    else:
        return 3

def draw_lanes(frame):
    h, w = frame.shape[:2]
    colors = [(0,200,255), (0,255,150), (255,200,0)]
    overlay = frame.copy()
    alpha = 0.15
    for i in range(3):
        x1, x2 = int(i*w/3), int((i+1)*w/3)
        cv2.rectangle(overlay, (x1,0), (x2,h), colors[i], -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    for i in range(1, 3):
        cv2.line(frame, (int(i*w/3),0), (int(i*w/3),h), (255,255,255), 2)
    return frame

# -------------------- Main --------------------
def main():
    video_path = "video.mp4"
    output_path = "output_annotated.mp4"
    csv_path = "counts.csv"
    model_path = "yolov8n.pt"

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_annotated.mp4', fourcc, 30, (frame_width, frame_height))

    count_line_y = int(height * 0.65)
    tracker = Sort()
    lane_counts = {1:0, 2:0, 3:0}
    counted_ids = set()
    csv_rows = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame = draw_lanes(frame)
        cv2.line(frame, (0, count_line_y), (width, count_line_y), (255,255,255), 2)

        results = model(frame, imgsz=640, conf=0.35, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls)]
            if cls_name in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf.cpu().numpy().item())
                dets.append([x1, y1, x2, y2, score])

        dets_np = np.array(dets) if len(dets) else np.empty((0,5))
        tracks = tracker.update(dets_np)

        for x1,y1,x2,y2,tid in tracks:
            cx, cy = int((x1+x2)//2), int((y1+y2)//2)
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,140,255), 2)
            cv2.putText(frame, f"ID {int(tid)}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)
            if cy >= count_line_y and int(tid) not in counted_ids:
                lane = lane_index_from_point(cx, width)
                lane_counts[lane] += 1
                counted_ids.add(int(tid))
                timestamp = round(frame_idx / fps, 3)
                csv_rows.append([int(tid), lane, frame_idx, timestamp])

        y0 = 30
        for lane in (1,2,3):
            cv2.putText(frame, f"Lane {lane}: {lane_counts[lane]}",
                        (10, y0 + 30*(lane-1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,255,50), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VehicleID","Lane","Frame","Timestamp(s)"])
        writer.writerows(csv_rows)

    print(f"Processing complete.\nOutput video: {output_path}\nCSV file: {csv_path}")
    print(f"Lane counts: {lane_counts}")

if __name__ == "__main__":
    main()
