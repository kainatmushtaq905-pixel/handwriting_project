import tkinter as tk
from tkinter import Canvas, ttk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition App")
        self.root.geometry("400x520")
        self.root.configure(bg="#2b2b2b")  # dark background

        # =======================
        #      STYLE
        # =======================
        style = ttk.Style()
        style.theme_use("clam")

        # Button style
        style.configure(
            "TButton",
            font=("Helvetica", 14, "bold"),
            padding=10,
            background="#4CAF50",
            foreground="white",
            borderwidth=0,
            focusthickness=3,
            relief="flat"
        )
        style.map(
            "TButton",
            background=[("active", "#45a049")],
            foreground=[("active", "white")]
        )

        # Label style
        style.configure(
            "TLabel",
            background="#2b2b2b",
            foreground="white",
            font=("Helvetica", 18)
        )

        # =======================
        #       CANVAS
        # =======================
        self.canvas_frame = tk.Frame(root, bg="#2b2b2b", bd=2, relief="sunken")
        self.canvas_frame.pack(pady=20)

        self.canvas = Canvas(self.canvas_frame, width=280, height=280, bg="white", bd=0, highlightthickness=3, highlightbackground="#4CAF50")
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons Frame
        self.btn_frame = tk.Frame(root, bg="#2b2b2b")
        self.btn_frame.pack(pady=15)

        self.predict_btn = ttk.Button(self.btn_frame, text="Predict", command=self.predict)
        self.predict_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = ttk.Button(self.btn_frame, text="Clear", command=self.clear)
        self.clear_btn.grid(row=0, column=1, padx=10)

        # Result Label
        self.label = ttk.Label(root, text="Draw a Digit")
        self.label.pack(pady=20)

    # Drawing
    def paint(self, event):
        x, y = event.x, event.y
        r = 12
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    # Preprocessing
    def preprocess(self):
        img = self.image.resize((28, 28))
        img = np.array(img)
        img = 255 - img  # invert (MNIST style)
        img = img / 255.0
        img = img.reshape(1, 28, 28)
        return img

    # Prediction
    def predict(self):
        img = self.preprocess()
        pred = model.predict(img)
        digit = np.argmax(pred)
        self.label.config(text=f"Prediction: {digit}")

    # Clear canvas
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label.config(text="Draw a Digit")


root = tk.Tk()
DigitApp(root)
root.mainloop()
