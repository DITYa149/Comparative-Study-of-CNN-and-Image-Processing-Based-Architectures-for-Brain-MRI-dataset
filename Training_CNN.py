 
import os
import cv2
import numpy as np
from tensorflow import keras
from cnn import BrainMRICNN

IMG_SIZE = 224

# Label maps
tumor_classes = ['Glioma', 'Meningioma', 'Pituitary', 'Normal', 'Other']
stroke_classes = ['Normal', 'Stroke']
ms_classes = ['Normal', 'MS']

def load_dataset(base_path):

    X = []
    y_tumor = []
    y_stroke = []
    y_ms = []

    # ---- NORMAL ----
    normal_path = os.path.join(base_path, "Normal")
    for file in os.listdir(normal_path):
        img = cv2.imread(os.path.join(normal_path, file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        X.append(img)
        y_tumor.append(3)   # Normal
        y_stroke.append(0)  # Normal
        y_ms.append(0)      # Normal

    # ---- TUMOR ----
    tumor_path = os.path.join(base_path, "Tumor")

    tumor_map = {
        "Glioma": 0,
        "Meningioma": 1,
        "Pituitary": 2
    }

    for t_type in tumor_map:
        folder = os.path.join(tumor_path, t_type)

        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

            X.append(img)
            y_tumor.append(tumor_map[t_type])
            y_stroke.append(0)
            y_ms.append(0)

    # ---- STROKE ----
    stroke_path = os.path.join(base_path, "stroke")

    for file in os.listdir(stroke_path):
        img = cv2.imread(os.path.join(stroke_path, file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        X.append(img)
        y_tumor.append(3)   # Normal
        y_stroke.append(1)  # Stroke
        y_ms.append(0)

    # ---- MS ----
    ms_path = os.path.join(base_path, "MS")

    for file in os.listdir(ms_path):
        img = cv2.imread(os.path.join(ms_path, file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

        X.append(img)
        y_tumor.append(3)
        y_stroke.append(0)
        y_ms.append(1)

    return (
        np.array(X),
        keras.utils.to_categorical(y_tumor, 5),
        keras.utils.to_categorical(y_stroke, 2),
        keras.utils.to_categorical(y_ms, 2)
    )


# ===== LOAD DATA =====
X, y_tumor, y_stroke, y_ms = load_dataset("dataset")

# ===== SPLIT =====
from sklearn.model_selection import train_test_split

X_train, X_val, yt_train, yt_val, ys_train, ys_val, ym_train, ym_val = train_test_split(
    X, y_tumor, y_stroke, y_ms, test_size=0.2, random_state=42
)

# ===== MODEL =====
cnn = BrainMRICNN()
cnn.build_model()
cnn.compile_model()

# ===== TRAIN =====
cnn.model.fit(
    X_train,
    {
        'brain_tumor': yt_train,
        'stroke': ys_train,
        'ms': ym_train
    },
    validation_data=(
        X_val,
        {
            'brain_tumor': yt_val,
            'stroke': ys_val,
            'ms': ym_val
        }
    ),
    epochs=20,
    batch_size=16
)

cnn.model.save("models/brain_mri_model.h5")
PREDICTION
import numpy as np
from tensorflow import keras
from image_preprocessing import ImagePreprocessor

class BrainMRIPredictor:

    def __init__(self, model_path='models/brain_mri_model.h5'):
        self.model = keras.models.load_model(model_path)
        self.preprocessor = ImagePreprocessor()

        self.tumor_classes = ['Glioma', 'Meningioma', 'Pituitary', 'Normal', 'Other']
        self.stroke_classes = ['Normal', 'Stroke']
        self.ms_classes = ['Normal', 'MS']

    def predict(self, image_path):

        image = self.preprocessor.preprocess(image_path)
        if image is None:
            return {'error': 'Image processing failed'}

        tumor_pred, stroke_pred, ms_pred = self.model.predict(image, verbose=0)

        predictions = {
            'tumor': self._get_result(tumor_pred, self.tumor_classes),
            'stroke': self._get_result(stroke_pred, self.stroke_classes),
            'ms': self._get_result(ms_pred, self.ms_classes)
        }

        # Symptom inference
        predictions['symptoms'] = self.infer_symptoms(predictions)

        # Health check
        predictions['healthy'] = (
            predictions['tumor']['class'] == 'Normal' and
            predictions['stroke']['class'] == 'Normal' and
            predictions['ms']['class'] == 'Normal'
        )

        return predictions

    def _get_result(self, pred, classes):
        return {
            'class': classes[np.argmax(pred[0])],
            'confidence': float(np.max(pred[0]))
        }

    def infer_symptoms(self, predictions):

        symptoms = []

        if predictions['tumor']['class'] != 'Normal':
            symptoms += ['Headache', 'Seizures']

        if predictions['stroke']['class'] == 'Stroke':
            symptoms += ['Memory Problems']

        if predictions['ms']['class'] == 'MS':
            symptoms += ['Memory Problems']

        return list(set(symptoms))

    # ✅ FIXED: Properly inside class
    def format_results(self, predictions):

        if 'error' in predictions:
            return f"Error: {predictions['error']}"

        result = "="*50 + "\n"
        result += "BRAIN MRI ANALYSIS RESULTS\n"
        result += "="*50 + "\n\n"

        # Tumor
        tumor = predictions['tumor']
        result += f"Tumor: {tumor['class']} ({tumor['confidence']:.2%})\n"

        # Stroke
        stroke = predictions['stroke']
        result += f"Stroke: {stroke['class']} ({stroke['confidence']:.2%})\n"

        # MS
        ms = predictions['ms']
        result += f"MS: {ms['class']} ({ms['confidence']:.2%})\n"

        # Symptoms
        symptoms = predictions['symptoms']
        result += f"Symptoms: {', '.join(symptoms) if symptoms else 'None'}\n"

        # Health
        result += f"\nOverall: {'HEALTHY' if predictions['healthy'] else 'ABNORMAL'}\n"

        return result