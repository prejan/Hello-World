import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import urllib.request
import os

# Download and save the labels file locally
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
label_filename = 'ImageNetLabels.txt'

if not os.path.exists(label_filename):
    urllib.request.urlretrieve(label_url, label_filename)

# Load the model
model = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')

# Load labels
with open(label_filename, 'r') as f:
    labels = f.read().strip().split('\n')

# Function to process image and get predictions
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.expand_dims(image, axis=0)
    results = model(image_np)
    result = {key:value.numpy() for key,value in results.items()}
    return result

# Function to draw bounding boxes and labels on the image
# Function to draw bounding boxes and labels on the image
MIN_SCORE = 0.5  # Adjust this threshold as needed

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, boxes, class_names, scores):
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        xmin = int(xmin * image.shape[1])
        xmax = int(xmax * image.shape[1])
        ymin = int(ymin * image.shape[0])
        ymax = int(ymax * image.shape[0])

        class_name = class_names[i]
        score = scores[i]

        if score >= MIN_SCORE:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, '{} {:.2f}'.format(class_name, score),
                        (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f'Detected: {class_name} with confidence {score:.2f}')

# Main function for processing webcam frames
def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Process the frame
        result = process_image(frame)
        boxes = result['detection_boxes'][0]
        class_indices = [int(idx) for idx in result['detection_classes'][0]]
        class_names = np.array([labels[idx] for idx in class_indices])
        scores = result['detection_scores'][0]

        # Draw boxes and labels
        draw_boxes(frame, boxes, class_names, scores)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
