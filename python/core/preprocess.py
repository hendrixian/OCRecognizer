import cv2
import numpy as np

class Preprocessor:
    """Preprocessing for NRC images and digit crops"""

    def __init__(self, target_height=32, target_width=128):
        self.target_height = target_height
        self.target_width = target_width

    def preprocess_full_image(self, image):
        """Preprocess full NRC image before YOLO detection"""
        denoised = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened

    def preprocess_for_crnn(self, image):
        """Preprocess concatenated digit image for CRNN input"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)

        corrected = cv2.absdiff(gray, background)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        denoised = cv2.fastNlMeansDenoising(corrected, h=10)

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            3
        )

        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        h, w = cleaned.shape
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cleaned, (new_w, new_h))

        canvas = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        y_off = (self.target_height - new_h) // 2
        x_off = (self.target_width - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        normalized = canvas.astype(np.float32) / 255.0
        return normalized
