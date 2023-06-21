from ultralytics import YOLO
import numpy

model = YOLO("yolov8n.pt")

model.train(data="data.yaml", epochs=30)
