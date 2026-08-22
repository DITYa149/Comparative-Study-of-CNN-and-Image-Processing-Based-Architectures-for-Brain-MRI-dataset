import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Load trained weights and normalization parameters
weights = np.load("trained_weights.npy")
feature_mean = np.load("feature_mean.npy")
feature_std = np.load("feature_std.npy")

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def extract_features_and_contour(image_path):

    original = cv2.imread(image_path)
    if original is None:
        return None, None, None
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    original = cv2.resize(original, (256, 256))

    gray = gray / 255.0
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE enhancement (same as training)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((blurred * 255).astype(np.uint8))
    
    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology - same as training
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    image_area = 256 * 256
    max_area = 0
    tumor_contour = None

    # Filter contours: keep only those between 5% and 80% of image area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.05 * image_area < area < 0.80 * image_area:
            if area > max_area:
                max_area = area
                tumor_contour = cnt

    # If no valid contour found, set max_area to minimum
    if max_area == 0:
        max_area = 0.001

    # Features (use original blurred image, not enhanced)
    area_feature = max_area / image_area
    mean_intensity = np.mean(blurred)
    std_intensity = np.std(blurred)

    left_half = blurred[:, :128]
    right_half = blurred[:, 128:]
    asymmetry = abs(np.mean(left_half) - np.mean(right_half))

    features = np.array([area_feature, mean_intensity,
                         std_intensity, asymmetry])

    return features, tumor_contour, original


def predict_and_display(image_path):
    
    result = extract_features_and_contour(image_path)
    
    if result[0] is None:
        result_label.config(text="Error: Could not load image!", fg="red")
        return
    
    features, tumor_contour, original = result

    # Normalize using TRAINING dataset statistics (crucial!)
    features = (features - feature_mean) / feature_std

    # Add bias
    features = np.insert(features, 0, 1)

    prediction = sigmoid(np.dot(features, weights))

    # Draw contour if tumor is detected
    if prediction >= 0.5:
        result_text = f"🧠 Prediction: TUMOR\nConfidence: {prediction:.4f}"
        result_label.config(text=result_text, fg="red")

        if tumor_contour is not None:
            cv2.drawContours(original, [tumor_contour], -1, (0, 0, 255), 2)
        else:
            result_label.config(text=result_text + "\n(No tumor contour detected)", fg="orange")

    else:
        result_text = f"✓ Prediction: NORMAL\nConfidence: {1-prediction:.4f}"
        result_label.config(text=result_text, fg="green")

    # Display the image with result
    display_image(original)


def display_image(cv_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Resize for display (fit in window)
    pil_image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    
    # Convert to PhotoImage
    photo = ImageTk.PhotoImage(pil_image)
    
    # Update label
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference


def upload_file():
    file_path = filedialog.askopenfilename(
        title="Select a Brain MRI Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
    )
    
    if file_path:
        file_label.config(text=f"File: {os.path.basename(file_path)}", fg="blue")
        predict_and_display(file_path)


# ----------- GUI SETUP -----------

root = tk.Tk()
root.title("Brain Tumor Detection System")
root.geometry("600x700")
root.config(bg="lightgray")

# Title
title_label = tk.Label(root, text="🧠 Brain Tumor Detection", 
                       font=("Arial", 18, "bold"), bg="lightgray")
title_label.pack(pady=10)

# Upload button
upload_button = tk.Button(root, text="📁 Upload MRI Image", 
                          command=upload_file, 
                          font=("Arial", 12, "bold"),
                          bg="skyblue", fg="black", padx=10, pady=10)
upload_button.pack(pady=10)

# File name label
file_label = tk.Label(root, text="No file selected", 
                      font=("Arial", 10), bg="lightgray", fg="gray")
file_label.pack(pady=5)

# Image display label
image_label = tk.Label(root, bg="white", width=500, height=400)
image_label.pack(pady=10, padx=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), 
                        bg="lightgray", justify="center")
result_label.pack(pady=10)

# Run GUI
root.mainloop()