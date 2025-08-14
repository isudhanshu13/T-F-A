# Vehicle-Counting-YOLO-SORT
Traffic Flow Analysis: Vehicle Counting per Lane
This project implements a real-time vehicle detection and tracking system using computer vision. It processes a video stream to detect vehicles, track them across frames, and count them in three distinct lanes. The system is built with a YOLOv8 object detection model and a SORT tracking algorithm for robust performance.

Core Features
Vehicle Detection: Utilizes a pre-trained YOLOv8 model to accurately detect cars, motorcycles, buses, and trucks in each video frame.

Lane Definition: Defines three virtual lanes across the video frame to categorize and count vehicles based on their position.

Vehicle Tracking: Employs the SORT (Simple Online and Realtime Tracking) algorithm to maintain a unique ID for each vehicle, preventing duplicate counts as they move through the lanes.

Real-time Processing: The script is optimized for smooth, near real-time execution on standard hardware.

Output Generation: Produces both a visual and a data-based output.

Visual: An annotated video with lane boundaries, bounding boxes around vehicles, and real-time lane counts.

CSV: A detailed .csv file logging each vehicle's ID, its lane, the frame count, and the timestamp when it was counted.

Requirements
The script requires the following Python libraries. It is highly recommended to use a virtual environment to manage dependencies.

opencv-python

ultralytics

filterpy

numpy

pandas

scipy
