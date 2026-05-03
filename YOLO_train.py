from ultralytics import YOLO

# Use pretrained YOLOv8 nano (smallest, fastest)
model = YOLO("yolov8n.pt")  # downloads automatically ✅

model.train(
    data="data/YOLO_format/data.yaml",
    epochs=50,
    batch=32,
    imgsz=96,        # our images are 96x96
    device=0,        # GPU ✅
    project="outputs/yolo",
    name="emotion_yolo_run_0",
)