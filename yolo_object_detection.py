# from ultralytics import YOLO

# # Load the base YOLOv8n model
# model = YOLO('yolov8n.pt')  

# # Train the model using your custom data
# results = model.train(
#     data=r'C:\Users\path\Desktop\jupyter_extension_prettier\vs_code_extension\Persian_Car_Plates_YOLOV8\data.yaml',      # Path to your data.yaml file
#     epochs=5,                          # Number of epochs
#     imgsz=640,                          # Image size
#     batch=16,                           # Batch size
#     patience=50,                        # Early stopping patience
#     save=True,                          # Save results                         
#     workers=8,                          # Number of worker threads
#     project='plates_detection',         # Project name
#     name='yolov8n_plates'              # Experiment name
# )





# from ultralytics import YOLO
# import cv2

# model = YOLO(r"C:\Users\Manthan\Desktop\jupyter_extension_prettier\vs_code_extension\plates_detection\yolov8n_plates\weights\best.pt")


# image_path = r"C:\Users\Manthan\Desktop\jupyter_extension_prettier\vs_code_extension\Persian_Car_Plates_YOLOV8\test\images\235_png.rf.96f2ccad5a0073336edfcccbb9cb06c3.jpg"

# image = cv2.imread(image_path)
# image = cv2.resize(image, (640, 640))  

# model.predict(source=image, conf=0.1)

# print(results)


# prompt: i want to visualise output that contains bounding box of detected class of test timage given above

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = YOLO(r"C:\Users\Manthan\Desktop\jupyter_extension_prettier\vs_code_extension\plates_detection\yolov8n_plates3\weights\best.pt")


image_path = r"C:\Users\Manthan\Desktop\jupyter_extension_prettier\vs_code_extension\Persian_Car_Plates_YOLOV8\test\images\235_png.rf.96f2ccad5a0073336edfcccbb9cb06c3.jpg"

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib


# Perform inference
model.predict(source=image, conf=0.1)


