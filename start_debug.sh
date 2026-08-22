#!/bin/sh
. venv/bin/activate
python hub.py --model models/yolov8n-face.onnx --device 0 --imgsz 640 --detect-fps 10 --fps 10 --delay-ms 200 --port 8443 --debug
