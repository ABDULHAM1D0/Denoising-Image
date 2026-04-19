import streamlit as st
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import easyocr
from qualityassessor import QualityAssessor
from unetskip import DiffusionUNet
import time
from datetime import datetime

print("Loaded documentprocessor from:", __file__)


class DocumentProcessor:
    def __init__(self, model_path="models/best_diffusion_model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_path = model_path
        self.ocr_reader = None
        self.quality_assessor = QualityAssessor()

    @st.cache_resource
    def load_model(_self):
        """Load the trained diffusion denoising model"""
        try:
            model = DiffusionUNet(in_channels=6).to(_self.device)  # 6 channels for conditional
            if os.path.exists(_self.model_path):
                # Load checkpoint
                checkpoint = torch.load(_self.model_path, map_location=_self.device)

                # Handle different save formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        st.success(
                            f"Diffusion model loaded from epoch {checkpoint.get('epoch', 'unknown')}, loss: {checkpoint.get('loss', 'N/A'):.4f}")
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

                model.eval()
                return model
            else:
                st.warning(f"Model not found at {_self.model_path}. Using untrained model for demo.")
                return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Tip: Your model is a Diffusion model. Make sure the architecture matches your training.")
            import traceback
            st.code(traceback.format_exc())
            return None

    @st.cache_resource
    def load_ocr(_self):
        """Load OCR reader"""
        try:
            reader = easyocr.Reader(['en'], gpu=(_self.device == "cuda"))
            return reader
        except Exception as e:
            st.error(f"Error loading OCR: {e}")
            return None

    def denoise_image(self, image, img_size=256):
        """Denoise the image using the trained diffusion model"""
        if self.model is None:
            self.model = self.load_model()

        if self.model is None:
            st.error("Model failed to load. Cannot denoise.")
            return image

        try:
            # Prepare image.
            img_resized = image.resize((img_size, img_size))
            img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(self.device)

            st.info(
                f"Input tensor - Shape: {img_tensor.shape}, Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")


            conditional_input = torch.cat([img_tensor, img_tensor], dim=1)

            # Try different timestep strategies
            timestep_strategy = st.session_state.get('timestep_strategy', 'zero')

            with torch.no_grad():
                if timestep_strategy == 'zero':
                    # Strategy 1: Use timestep=0 (what we were doing)
                    timestep = torch.zeros(1, device=self.device)
                elif timestep_strategy == 'middle':
                    # Strategy 2: Use middle timestep
                    timestep = torch.tensor([500.0], device=self.device)
                else:  # 'high'
                    # Strategy 3: Use high timestep
                    timestep = torch.tensor([999.0], device=self.device)

                denoised = self.model(conditional_input, timestep)

                st.info(f"Raw output - Range: [{denoised.min():.3f}, {denoised.max():.3f}]")


                denoised_min = denoised.min()
                denoised_max = denoised.max()

                if denoised_max - denoised_min > 0:
                    # Normalize to [0, 1].
                    denoised = (denoised - denoised_min) / (denoised_max - denoised_min)
                    st.success(f"Normalized output to range [0, 1]")
                else:
                    st.warning("Output has no variation, using clamping instead")
                    denoised = torch.clamp(denoised, 0, 1)

                st.info(f"After normalization - Range: [{denoised.min():.3f}, {denoised.max():.3f}]")

                if denoised.shape[-2:] != (img_size, img_size):
                    denoised = F.interpolate(denoised, size=(img_size, img_size), mode='bilinear', align_corners=False)

            denoised_np = (denoised.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            st.info(
                f"NumPy array - Shape: {denoised_np.shape}, Range: [{denoised_np.min()}, {denoised_np.max()}], Mean: {denoised_np.mean():.1f}")

            # Apply contrast enhancement if image is too dark or washed out
            mean_brightness = denoised_np.mean()
            if mean_brightness < 50 or mean_brightness > 200:
                st.warning(f"Unusual brightness (mean={mean_brightness:.1f}), applying auto-contrast...")

                # Convert to PIL for processing
                denoised_pil_temp = Image.fromarray(denoised_np)

                # Apply histogram equalization for better contrast
                from PIL import ImageOps
                denoised_pil_temp = ImageOps.autocontrast(denoised_pil_temp, cutoff=2)
                denoised_np = np.array(denoised_pil_temp)

                st.success(f"Contrast enhanced - New mean: {denoised_np.mean():.1f}")

            # Check if output is blank
            if denoised_np.std() < 5:  # Very low variation = likely blank
                st.warning("Denoised output appears blank/uniform! Using original image instead.")
                return image

            # Resize back to original size if needed
            if image.size != (img_size, img_size):
                denoised_pil = Image.fromarray(denoised_np).resize(image.size, Image.LANCZOS)
            else:
                denoised_pil = Image.fromarray(denoised_np)

            # Final check: ensure it's not completely washed out
            denoised_np_check = np.array(denoised_pil)
            if denoised_np_check.std() < 10:
                st.warning("Denoised result has very low contrast. Using original image.")
                return image

            return denoised_pil

        except Exception as e:
            st.error(f"Error during denoising: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.warning("Falling back to original image")
            return image

    def extract_text(self, image):
        """Extract text using OCR"""
        if self.ocr_reader is None:
            self.ocr_reader = self.load_ocr()

        img_np = np.array(image)
        try:
            results = self.ocr_reader.readtext(img_np)
            text_lines = [result[1] for result in results]
            full_text = "\n".join(text_lines)

            # Also return structured results
            structured_results = [
                {
                    "text": result[1],
                    "confidence": float(result[2]),
                    "bbox": result[0]
                }
                for result in results
            ]

            return full_text, structured_results
        except Exception as e:
            st.error(f"OCR Error: {e}")
            return "", []

    def extract_structured_data(self, text, doc_type="invoice"):
        """Extract structured fields from text (simple pattern matching)"""
        import re

        extracted = {
            "document_type": doc_type,
            "raw_text": text,
            "fields": {}
        }

        if doc_type == "invoice":
            # Invoice number
            inv_match = re.search(r'(?:invoice|inv)[#:\s]*([A-Z0-9-]+)', text, re.IGNORECASE)
            if inv_match:
                extracted["fields"]["invoice_number"] = inv_match.group(1)

            # Date
            date_match = re.search(r'(?:date|dated)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
            if date_match:
                extracted["fields"]["date"] = date_match.group(1)

            # Total
            total_match = re.search(r'(?:total|amount)[:\s]*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
            if total_match:
                extracted["fields"]["total"] = total_match.group(1)

            # Vendor/Company
            lines = text.split('\n')
            if lines:
                extracted["fields"]["vendor"] = lines[0].strip()

        elif doc_type == "receipt":
            # Merchant
            lines = text.split('\n')
            if lines:
                extracted["fields"]["merchant"] = lines[0].strip()

            # Date
            date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
            if date_match:
                extracted["fields"]["date"] = date_match.group(1)

            # Total
            total_match = re.search(r'(?:total|amount)[:\s]*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
            if total_match:
                extracted["fields"]["total"] = total_match.group(1)

        return extracted

    def process_document(self, image, doc_type="invoice", denoise=True):
        """Complete processing pipeline"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "document_type": doc_type,
                "denoising_enabled": denoise
            }
        }

        quality = self.quality_assessor.assess_quality(image)
        results["quality_assessment"] = quality

        if denoise:
            start = time.time()
            denoised_image = self.denoise_image(image)
            results["denoising_time"] = time.time() - start
            processing_image = denoised_image
        else:
            denoised_image = None
            processing_image = image

        start = time.time()
        text, ocr_results = self.extract_text(processing_image)
        results["ocr_time"] = time.time() - start
        results["extracted_text"] = text
        results["ocr_results"] = ocr_results

        start = time.time()
        structured = self.extract_structured_data(text, doc_type)
        results["extraction_time"] = time.time() - start
        results["structured_data"] = structured

        if denoise:
            orig_text, _ = self.extract_text(image)
            results["original_text"] = orig_text

            # Calculate improvement
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, orig_text, text).ratio()
            results["ocr_improvement"] = similarity

        return results, denoised_image
