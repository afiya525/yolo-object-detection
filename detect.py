from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Save results
results[0].save(filename="results.jpg")

print("Detection completed! Image saved as results.jpg")