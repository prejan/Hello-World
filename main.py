import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_and_display(img_path):
    # Redirect standard output to suppress progress bar output
    sys.stdout = open(os.devnull, 'w')
    model = MobileNet(weights='imagenet')
    sys.stdout = sys.__stdout__
    
    img_array = load_and_preprocess_image(img_path)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    title = "\n".join([f"{pred[1]}: {pred[2]*100:.2f}%" for pred in decoded_predictions])
    
    # Display the image and prediction in the Tkinter window
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = ImageTk.PhotoImage(img)
    
    image_label.config(image=img)
    image_label.image = img
    prediction_label.config(text=title)

def select_image():
    img_path = filedialog.askopenfilename()
    if img_path:
        predict_and_display(img_path)

# Initialize Tkinter window
root = tk.Tk()
root.title("Image Recognizer")

# Create and place the "Select Image" button
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# Create and place the label for displaying the image
image_label = tk.Label(root)
image_label.pack()

# Create and place the label for displaying predictions
prediction_label = tk.Label(root, text="", wraplength=300)
prediction_label.pack()

# Run the Tkinter event loop
root.mainloop()
