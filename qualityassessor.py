from PIL import Image
import cv2
import numpy as np


class QualityAssessor:
    @staticmethod
    def assess_quality(image):
        """Assess document quality"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Compute quality metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_estimate = np.std(gray)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # Calculate quality score (0-1)
        quality_score = (
                min(laplacian_var / 500, 1.0) * 0.3 +
                max(0, 1 - noise_estimate / 50) * 0.3 +
                (1 - abs(brightness - 128) / 128) * 0.2 +
                min(contrast / 50, 1.0) * 0.2
        )

        return {
            "score": quality_score,
            "blur": laplacian_var,
            "noise": noise_estimate,
            "brightness": brightness,
            "contrast": contrast,
            "status": "Good" if quality_score > 0.7 else "Moderate" if quality_score > 0.4 else "Poor"
        }
