import os
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    import keras
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk

MODEL_DIR = "saved_model"
model_path_h5 = os.path.join(MODEL_DIR, "final_model.h5")
model_path_keras = os.path.join(MODEL_DIR, "final_model.keras")

# Load model (prefer .keras if exists)
model = None
if os.path.exists(model_path_h5):
    try:
        model = keras.models.load_model(model_path_h5)
    except Exception as e:
        print("Failed loading .h5 model:", e)
if model is None and os.path.exists(model_path_keras):
    try:
        model = keras.models.load_model(model_path_keras)
    except Exception as e:
        print("Failed loading .keras model:", e)

if model is None:
    raise FileNotFoundError("No trained model found. Run train_model.py first to generate saved_model/final_model.h5 or final_model.keras")

# GUI
root = tk.Tk()
root.title("Handwritten Digit Recognition - GUI")
root.resizable(False, False)

canvas_width, canvas_height = 280, 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

image1 = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image1)

pen_size = tk.IntVar(value=8)

def paint(event):
    size = pen_size.get()
    x1, y1 = (event.x - size), (event.y - size)
    x2, y2 = (event.x + size), (event.y + size)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=size)
    draw.ellipse([x1, y1, x2, y2], fill=0)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)

def predict_digit():
    img_resized = image1.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized)
    img_array = np.array(img_inverted).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    pred = model.predict(img_array)
    digit = int(np.argmax(pred))
    prob = float(np.max(pred) * 100.0)
    result_label.config(text=f"Prediction: {digit} ({prob:.2f}%)")

predict_btn = tk.Button(root, text="Predict", command=predict_digit)
predict_btn.grid(row=1, column=0, pady=10)

clear_btn = tk.Button(root, text="Clear", command=clear_canvas)
clear_btn.grid(row=1, column=1, pady=10)

result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Arial", 14))
result_label.grid(row=1, column=2, columnspan=2)

pen_slider = tk.Scale(root, from_=2, to=20, orient="horizontal", label="Pen Size", variable=pen_size)
pen_slider.grid(row=2, column=0, columnspan=4, pady=10)

canvas.bind("<B1-Motion>", paint)

root.mainloop()