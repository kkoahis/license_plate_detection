from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(source='./videos/sample3.mp4', show=True, classes=[2,3,5,7])
