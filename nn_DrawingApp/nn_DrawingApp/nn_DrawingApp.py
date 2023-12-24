import tkinter as tk
from tkinter import filedialog
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import sys
import os

from network import NeuralNetwork

def load_network():
    file_path = filedialog.askopenfilename(title="Select network data file", filetypes=[("NPZ files", "*.npz")])
    nn = NeuralNetwork.load(file_path)
    return nn


class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")


        self.canvas = Canvas(master, width=280, height=280, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.master.bind("<Key>", self.key_press)

        self.prediction_label = tk.Label(master, text="Prediction: ")
        self.prediction_label.pack()

        # List to store percentage predictions for each digit
        self.percentage_labels = []
        for digit in range(10):
            label = tk.Label(master, text=f"{digit}: 0.00%", fg="blue")
            label.pack()
            self.percentage_labels.append(label)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=23)
        self.draw.line([x1, y1, x2, y2], fill="black", width=23)

        # Predict whenever the picture changes
        self.predict_digit()

    def key_press(self, event):
        if event.char.lower() == 'r':
            self.reset()

    def reset(self):
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")  # Clear the canvas

    def predict_digit(self):
        # Resize the image to 28x28 and invert 
        resized_image = ImageOps.invert(self.image.resize((28, 28), Image.LANCZOS))

        # Convert the image to a NumPy array and normalize
        input_data = np.array(resized_image).reshape(28 * 28) / 255.0

        # Forward propagate through the neural network
        prediction = nn.forward_prop(input_data)

        # Display the predicted digit
        predicted_digit = np.argmax(prediction)
        self.prediction_label.config(text=f"Prediction: {predicted_digit}")

        total_sum = np.sum(prediction)
        # Display percentage predictions for each digit
        percentages = [f"{digit}: {float(prediction[digit][0]/total_sum*100):.2f}%" for digit in sorted(range(10), key=lambda x: prediction[x][0], reverse=True)]

        for label, percentage in zip(self.percentage_labels, percentages):
            label.config(text=percentage)

        # Update color based on the predicted digit
        for label in self.percentage_labels:
            label.config(fg="black")
        self.percentage_labels[0].config(fg="blue")

if __name__ == "__main__":
    # Call the function to load the network
    nn = load_network()
    root = tk.Tk()
    
    app = DrawingApp(root)
    root.mainloop()
