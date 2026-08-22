import cv2
import numpy as np

class ImagePreprocessor:

    def __init__(self, target_size=(224, 224), use_zscore=False):
        self.target_size = target_size
        self.use_zscore = use_zscore

    def load_image(self, image_path):
        try:
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def apply_gaussian_blur(self, image, kernel_size=(5, 5), sigma=1.0):
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def resize_image(self, image):
        return cv2.resize(image, self.target_size)

    def normalize_image(self, image):
        image = image.astype(np.float32)

        if self.use_zscore:
            mean = np.mean(image)
            std = np.std(image) + 1e-8
            image = (image - mean) / std
        else:
            image = image / 255.0

        return image

    def preprocess(self, image_path):
        image = self.load_image(image_path)
        if image is None:
            return None

        image = self.apply_gaussian_blur(image)
        image = self.resize_image(image)
        image = self.normalize_image(image)

        image = np.expand_dims(image, axis=0)
        return image