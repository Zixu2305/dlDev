import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os
import json
IMGSIZE = 224
with open('labels.json', 'r') as f:
    labels = json.load(f)

model = tf.keras.models.load_model('best_model.keras')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

x_start, y_start, width, height = 100, 100, 500, 500  # Example: center region of the frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Draw the rectangle for the region of interest (ROI) on the frame
    cv2.rectangle(frame, (x_start, y_start), (x_start + width, y_start + height), (0, 255, 0), 2)

    # Extract the region of interest (ROI)
    roi = frame[y_start:y_start + height, x_start:x_start + width]

    inputSize = (IMGSIZE, IMGSIZE)
    img = cv2.resize(roi, inputSize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize if required by your model
    # Make predictions
    predictions = model.predict(img)
    predicted_label = labels[np.argmax(predictions)]
    
    # Annotate the frame with the predicted label
    cv2.putText(frame, f'Predicted: {predicted_label}', 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    cv2.imshow('ROI', roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
