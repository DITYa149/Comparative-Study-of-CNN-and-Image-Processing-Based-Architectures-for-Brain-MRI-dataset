import cv2
import numpy as np
import os
import csv

# ----------- FEATURE EXTRACTION FUNCTION -----------

def extract_features(image_path):

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to fixed size
    img = cv2.resize(img, (256, 256))

    # Normalize
    img = img / 255.0

    # Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu Threshold
    _, thresh = cv2.threshold((blurred*255).astype(np.uint8), 
                              0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = 256 * 256
    max_area = 0

    # Get largest region
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area

    # Normalize area
    area_feature = max_area / image_area

    # Intensity Features
    mean_intensity = np.mean(blurred)
    std_intensity = np.std(blurred)

    # Asymmetry Feature
    left_half = blurred[:, :128]
    right_half = blurred[:, 128:]
    asymmetry = abs(np.mean(left_half) - np.mean(right_half))

    return [area_feature, mean_intensity, std_intensity, asymmetry]


# ----------- DATASET PROCESSING -----------

def process_dataset(dataset_path, output_csv="features.csv"):

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["area", "mean", "std", "asymmetry", "label"])

        for label_name in ["normal", "tumor"]:
            folder = os.path.join(dataset_path, label_name)
            label = 0 if label_name == "normal" else 1

            for filename in os.listdir(folder):
                image_path = os.path.join(folder, filename)
                features = extract_features(image_path)

                if features is not None:
                    writer.writerow(features + [label])

    print("Feature extraction complete. Saved to features.csv")


# ----------- RUN -----------

if __name__ == "__main__":
    dataset_path = "dataset"   # folder containing normal/ and tumor/
    process_dataset(dataset_path)