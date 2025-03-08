import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the waste classification model
model = load_model('waste_classification_model.h5')

# Define the classes
classes = ['Biodegradable', 'Non-Biodegradable']

# Verify YOLOv3 files exist
yolo_weights = 'yolov3.weights'
yolo_cfg = 'yolov3.cfg'
yolo_names = 'coco.names'  

if not os.path.exists(yolo_weights) or not os.path.exists(yolo_cfg) or not os.path.exists(yolo_names):
    raise FileNotFoundError("Missing YOLOv3 files. Ensure yolov3.weights, yolov3.cfg, and coco.names exist.")

# Load YOLOv3 model
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Load COCO class labels
with open(yolo_names, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcamq
cap = cv2.VideoCapture(0)

# Maximize the output window
cv2.namedWindow('Waste Classification', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Waste Classification', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Detect objects using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to store detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out low-confidence detections & exclude persons (class_id 0)
            if confidence > 0.6 and class_labels[class_id] != "person":
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.5)

    # Loop over the filtered boxes
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i  # Handle single index
        box = boxes[i]
        x, y, w, h = box

        # Ensure the bounding box is within the frame dimensions
        if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height:
            # Extract the region of interest (ROI)
            roi = frame[y:y + h, x:x + w]

            # Check if ROI is valid
            if roi.size > 0:
                # Preprocess the ROI for waste classification
                resized_roi = cv2.resize(roi, (64, 64))  # Resize to match model input size
                normalized_roi = resized_roi / 255.0  # Normalize pixel values
                input_roi = np.expand_dims(normalized_roi, axis=0)  # Add batch dimension

                # Predict the class
                prediction = model.predict(input_roi)
                confidence_score = float(prediction[0][0])  # Get confidence score

                # Set class based on prediction
                if confidence_score < 0.5:
                    predicted_class = "Biodegradable"
                    confidence_display = 1 - confidence_score
                    color = (0, 255, 0)  # Green for biodegradable
                else:
                    predicted_class = "Non-Biodegradable"
                    confidence_display = confidence_score
                    color = (0, 0, 255)  # Red for non-biodegradable

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{predicted_class} ({confidence_display:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Waste Classification', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
