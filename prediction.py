# ✅ NEW: CNN_Prediction.py - Complete prediction with GUI
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from tensorflow import keras
from image_preprocessing import ImagePreprocessor

class BrainMRIPredictor:
    """
     COMPLETE: Multi-task CNN predictor with inference
    """

    def __init__(self, model_path='models/brain_mri_model.h5'):
        """Load trained model and preprocessor"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            raise FileNotFoundError(f"❌ Model not found: {model_path}\n{e}")

        self.preprocessor = ImagePreprocessor(use_zscore=False, apply_clahe=True)

        # Class labels
        self.tumor_classes = ['Glioma', 'Meningioma', 'Pituitary', 'Normal', 'Other']
        self.stroke_classes = ['Normal', 'Stroke']
        self.ms_classes = ['Normal', 'MS']

    def predict(self, image_path):
        """
        Predict all three tasks for given image
        ✅ FIX: Complete error handling
        """
        try:
            # Preprocess image
            image = self.preprocessor.preprocess(image_path)
            if image is None:
                return {'error': 'Image processing failed'}

            # Get predictions from all three task heads
            tumor_pred, stroke_pred, ms_pred = self.model.predict(image, verbose=0)

            predictions = {
                'tumor': self._get_result(tumor_pred, self.tumor_classes),
                'stroke': self._get_result(stroke_pred, self.stroke_classes),
                'ms': self._get_result(ms_pred, self.ms_classes)
            }

            # Infer symptoms based on predictions
            predictions['symptoms'] = self.infer_symptoms(predictions)

            # Overall health status
            predictions['healthy'] = (
                predictions['tumor']['class'] == 'Normal' and
                predictions['stroke']['class'] == 'Normal' and
                predictions['ms']['class'] == 'Normal'
            )

            return predictions

        except Exception as e:
            return {'error': str(e)}

    def _get_result(self, pred, classes):
        """Extract class and confidence from prediction"""
        class_idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0]))
        
        return {
            'class': classes[class_idx],
            'confidence': confidence,
            'all_scores': {classes[i]: float(pred[0][i]) for i in range(len(classes))}
        }

    def infer_symptoms(self, predictions):
        """
        ✅ FIX: Improved symptom inference logic
        """
        symptoms = []

        # Tumor-related symptoms
        if predictions['tumor']['class'] != 'Normal':
            if predictions['tumor']['class'] in ['Glioma', 'Meningioma']:
                symptoms.extend(['Headache', 'Seizures', 'Vision Problems'])
            elif predictions['tumor']['class'] == 'Pituitary':
                symptoms.extend(['Headache', 'Hormonal Imbalance'])

        # Stroke-related symptoms
        if predictions['stroke']['class'] == 'Stroke':
            symptoms.extend(['Weakness', 'Memory Problems', 'Speech Difficulty'])

        # MS-related symptoms
        if predictions['ms']['class'] == 'MS':
            symptoms.extend(['Numbness', 'Fatigue', 'Vision Issues', 'Balance Problems'])

        # Remove duplicates and sort
        return sorted(list(set(symptoms)))

    def format_results(self, predictions):
        """Format predictions for display"""
        if 'error' in predictions:
            return f"❌ Error: {predictions['error']}"

        result = "=" * 60 + "\n"
        result += "🧠 BRAIN MRI ANALYSIS RESULTS\n"
        result += "=" * 60 + "\n\n"

        # Tumor results
        tumor = predictions['tumor']
        result += f"🔬 TUMOR CLASSIFICATION\n"
        result += f"   Class: {tumor['class']}\n"
        result += f"   Confidence: {tumor['confidence']:.2%}\n\n"

        # Stroke results
        stroke = predictions['stroke']
        result += f"⚡ STROKE DETECTION\n"
        result += f"   Class: {stroke['class']}\n"
        result += f"   Confidence: {stroke['confidence']:.2%}\n\n"

        # MS results
        ms = predictions['ms']
        result += f"🔗 MULTIPLE SCLEROSIS\n"
        result += f"   Class: {ms['class']}\n"
        result += f"   Confidence: {ms['confidence']:.2%}\n\n"

        # Symptoms
        symptoms = predictions['symptoms']
        result += f"⚕️ PREDICTED SYMPTOMS\n"
        if symptoms:
            for symptom in symptoms:
                result += f"   • {symptom}\n"
        else:
            result += "   None detected\n"

        # Overall status
        result += "\n" + "=" * 60 + "\n"
        status = "✅ HEALTHY" if predictions['healthy'] else "⚠️ ABNORMAL"
        result += f"OVERALL STATUS: {status}\n"
        result += "=" * 60 + "\n"

        return result


# ===== GUI APPLICATION =====
class BrainMRIGUI:
    """
    ✅ NEW: Complete GUI for CNN predictions
    """

    def __init__(self, root, model_path='models/brain_mri_model.h5'):
        self.root = root
        self.root.title("🧠 Brain MRI Analysis System - CNN Model")
        self.root.geometry("900x1000")
        self.root.config(bg="#f0f0f0")

        try:
            self.predictor = BrainMRIPredictor(model_path)
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.root.destroy()
            return

        self.current_image_path = None
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components"""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50")
        header_frame.pack(fill="x", padx=0, pady=0)

        title = tk.Label(header_frame, text="🧠 Brain MRI Analysis System",
                        font=("Arial", 20, "bold"), fg="white", bg="#2c3e50")
        title.pack(pady=15)

        subtitle = tk.Label(header_frame, text="Deep Learning CNN Model - Multi-Task Analysis",
                           font=("Arial", 10, "italic"), fg="#ecf0f1", bg="#2c3e50")
        subtitle.pack(pady=5)

        # Upload button
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=20)

        upload_btn = tk.Button(button_frame, text="📁 Upload MRI Image",
                              command=self.upload_file,
                              font=("Arial", 12, "bold"),
                              bg="#3498db", fg="white",
                              padx=20, pady=10,
                              relief="raised", bd=2)
        upload_btn.pack(side="left", padx=10)

        clear_btn = tk.Button(button_frame, text="🔄 Clear",
                             command=self.clear_display,
                             font=("Arial", 12, "bold"),
                             bg="#95a5a6", fg="white",
                             padx=20, pady=10,
                             relief="raised", bd=2)
        clear_btn.pack(side="left", padx=10)

        # File name label
        self.file_label = tk.Label(self.root, text="No file selected",
                                   font=("Arial", 10), bg="#f0f0f0", fg="gray")
        self.file_label.pack(pady=5)

        # Image display
        self.image_label = tk.Label(self.root, bg="white", relief="sunken", bd=2)
        self.image_label.pack(pady=10, padx=15, fill="both", expand=False)

        # Results display
        result_frame = tk.Frame(self.root, bg="#ecf0f1", relief="sunken", bd=2)
        result_frame.pack(pady=10, padx=15, fill="both", expand=True)

        tk.Label(result_frame, text="ANALYSIS RESULTS",
                font=("Arial", 11, "bold"), bg="#ecf0f1", fg="#2c3e50").pack(anchor="w", padx=10, pady=5)

        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side="right", fill="y")

        self.result_text = tk.Text(result_frame, height=15, width=80,
                                   font=("Courier", 10),
                                   yscrollcommand=scrollbar.set,
                                   bg="white", fg="#2c3e50")
        self.result_text.pack(padx=10, pady=10, fill="both", expand=True)
        scrollbar.config(command=self.result_text.yview)

    def upload_file(self):
        """Handle file upload"""
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                      ("All files", "*.*")]
        )

        if file_path:
            self.current_image_path = file_path
            self.file_label.config(text=f"File: {os.path.basename(file_path)}", fg="blue")
            self.analyze_image()

    def analyze_image(self):
        """Analyze selected image"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return

        try:
            # Get predictions
            predictions = self.predictor.predict(self.current_image_path)

            # Display image
            self.display_image(self.current_image_path)

            # Display results
            results_text = self.predictor.format_results(predictions)
            self.result_text.config(state="normal")
            self.result_text.delete(1.0, "end")
            self.result_text.insert(1.0, results_text)
            self.result_text.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def display_image(self, image_path):
        """Display image in GUI"""
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (400, 400))

            pil_img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(pil_img)

            self.image_label.config(image=photo, width=400, height=400)
            self.image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {e}")

    def clear_display(self):
        """Clear all displays"""
        self.current_image_path = None
        self.file_label.config(text="No file selected", fg="gray")
        self.image_label.config(image="", text="Image will appear here")
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, "end")
        self.result_text.config(state="disabled")


# ===== MAIN =====
if __name__ == "__main__":
    root = tk.Tk()
    app = BrainMRIGUI(root, model_path='models/brain_mri_model.h5')
    root.mainloop()