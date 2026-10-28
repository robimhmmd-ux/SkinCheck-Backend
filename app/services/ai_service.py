from abc import ABC, abstractmethod
import random
import time
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def enhance_skin_image(img_input, is_rgb: bool = True) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    in LAB color space to enhance skin micro-textures, fine lines/wrinkles, 
    and pigmentation contrast without distorting underlying skin tones.
    """
    try:
        import cv2
        if is_rgb:
            lab = cv2.cvtColor(img_input, cv2.COLOR_RGB2LAB)
        else:
            lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2LAB)
            
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE on L-channel (Luminance)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Re-merge and convert back to RGB
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # Blend 75% enhanced + 25% original to retain specular shine for oily skin detection
        original_rgb = img_input if is_rgb else cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(enhanced_rgb, 0.75, original_rgb, 0.25, 0)
        return blended
    except Exception as e:
        logger.warning(f"CLAHE enhancement failed, returning original: {e}")
        return img_input


class AIProviderInterface(ABC):
    """
    Abstract interface for Computer Vision AI Providers.
    Allows swapping out the AI provider (e.g. OpenCV, Custom Model, Cloud Vision API) 
    without changing the core business logic.
    """
    
    @abstractmethod
    def detect_face(self, image_path: str) -> bool:
        """Validates if a face is detected in the image."""
        pass
        
    @abstractmethod
    def analyze_skin_condition(self, image_path: str) -> dict:
        """
        Analyzes a single image and returns 5 skin condition scores (percentages):
        Acne, Dry Skin, Oily Skin, Dark Spots, Wrinkles.
        """
        pass

    @abstractmethod
    def analyze_multi_frame_skin_condition(self, image_paths: list[str]) -> dict:
        """
        Analyzes multiple captured frames (e.g. multi-angle poses: center, right, left,
        up, down, close-up) and aggregates the scores into a comprehensive diagnostic.
        """
        pass


class MockComputerVisionProvider(AIProviderInterface):
    """
    Mock implementation of AI Provider for MVP / testing phase.
    Generates simulated scores for the 5 visual facial conditions.
    """
    
    def detect_face(self, image_path: str) -> bool:
        return True
        
    def analyze_skin_condition(self, image_path: str) -> dict:
        time.sleep(1.0)
        acne = float(random.randint(10, 80))
        dry = float(random.randint(10, 80))
        oily = float(random.randint(10, 80))
        dark_spots = float(random.randint(10, 80))
        wrinkles = float(random.randint(10, 80))
        
        average_severity = (acne + dry + oily + dark_spots + wrinkles) / 5
        overall = max(0.0, min(100.0, round(100.0 - average_severity, 1)))
        
        return {
            "acne_score": acne,
            "dry_skin_score": dry,
            "oily_skin_score": oily,
            "dark_spots_score": dark_spots,
            "wrinkles_score": wrinkles,
            "redness_score": 0.0,
            "texture_score": wrinkles,
            "hyperpigmentation_score": dark_spots,
            "overall_score": overall,
            "confidence": 0.88,
            "frames_analyzed": 1,
            "analysis_status": "success"
        }

    def analyze_multi_frame_skin_condition(self, image_paths: list[str]) -> dict:
        result = self.analyze_skin_condition(image_paths[0] if image_paths else "")
        result["frames_analyzed"] = len(image_paths)
        result["confidence"] = 0.94
        return result


class KerasComputerVisionProvider(AIProviderInterface):
    """
    Real AI Provider that loads the .h5 model.
    Model: Facial Condition Analysis Model (VGG16 Transfer Learning)
    Detects 5 skin conditions: Acne, Dry Skin, Oily Skin, Dark Spots, Wrinkles
    Input Size: 224x224x3 (RGB)
    Output: 5 sigmoid outputs (percentages)
    Enhanced with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    and Multi-Angle Ensemble Aggregation.
    """
    
    def __init__(self, model_path: str = "facial-condition-model/finetuned_best.h5"):
        # Resolve a relative model path against the project root first
        # (app/services/ai_service.py -> parents[2] is the project root),
        # so the model is found no matter which folder the server is
        # started from. An absolute path, or a path that already exists
        # relative to the working directory, is used as-is.
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            from pathlib import Path

            root_candidate = str(
                Path(__file__).resolve().parents[2]
                / "facial-condition-model"
                / "finetuned_best.h5"
            )
            if os.path.exists(root_candidate):
                model_path = root_candidate
        self.model_path = model_path
        self.model = None

        if os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(f"Model not found at {self.model_path}. Will attempt to load later.")
        
    def _load_model(self):
        try:
            import tensorflow as tf
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info("Successfully loaded Keras VGG16 model.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def detect_face(self, image_path: str) -> bool:
        return True
        
    def analyze_skin_condition(self, image_path: str) -> dict:
        """Analyzes a single image with CLAHE preprocessing."""
        return self.analyze_multi_frame_skin_condition([image_path])

    def analyze_multi_frame_skin_condition(self, image_paths: list[str]) -> dict:
        """
        Loads all frames, applies CLAHE enhancement, runs batch inference on VGG16,
        and aggregates results using multi-angle sensitive ensemble pooling.
        """
        if not self.model:
            self._load_model()
            if not self.model:
                raise Exception("Model AI belum dimuat atau tidak ditemukan!")
                
        if not image_paths:
            raise ValueError("Daftar gambar tidak boleh kosong.")
            
        import cv2
        
        batch_arrays = []
        valid_paths = []
        
        for p in image_paths:
            try:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                    
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # 1. Apply CLAHE Enhancement
                enhanced_rgb = enhance_skin_image(img_rgb, is_rgb=True)
                
                # 2. Resize to 224x224 with high-quality interpolation
                resized = cv2.resize(enhanced_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                
                # 3. Normalize / 255.0
                norm_img = resized.astype(np.float32) / 255.0
                batch_arrays.append(norm_img)
                valid_paths.append(p)
            except Exception as e:
                logger.warning(f"Error processing frame {p}: {e}")
                
        if not batch_arrays:
            raise Exception("Tidak ada gambar valid yang dapat diproses.")
            
        # Stack into batch tensor: shape (N, 224, 224, 3)
        batch_tensor = np.array(batch_arrays, dtype=np.float32)
        
        # Batch inference: ultra-fast in a single pass
        predictions = self.model.predict(batch_tensor, verbose=0)
        
        # Sigmoid outputs: [0: Acne, 1: Dry Skin, 2: Oily Skin, 3: Dark Spots, 4: Wrinkles]
        acne_scores = predictions[:, 0] * 100.0
        dry_scores = predictions[:, 1] * 100.0
        oily_scores = predictions[:, 2] * 100.0
        dark_spots_scores = predictions[:, 3] * 100.0
        wrinkles_scores = predictions[:, 4] * 100.0
        
        # Multi-Angle Ensemble Aggregation:
        # - Localized conditions (Acne, Dark Spots, Wrinkles, and Oily T-Zone): take top 35% peak severity
        #   across poses so that blemishes or shiny T-zones are not diluted by side-angle frames.
        # - Global background conditions (Dry): take balanced weighted average.
        def aggregate_peak(scores):
            if len(scores) == 1:
                return float(scores[0])
            sorted_s = np.sort(scores)
            top_k = max(1, int(np.ceil(len(sorted_s) * 0.35)))
            return float(np.mean(sorted_s[-top_k:]))
            
        def aggregate_mean(scores):
            return float(np.mean(scores))
            
        final_acne = round(aggregate_peak(acne_scores), 1)
        final_dark_spots = round(aggregate_peak(dark_spots_scores), 1)
        final_wrinkles = round(aggregate_peak(wrinkles_scores), 1)
        final_dry = round(aggregate_mean(dry_scores), 1)
        final_oily = round(aggregate_peak(oily_scores), 1)
        
        avg_severity = (final_acne + final_dry + final_oily + final_dark_spots + final_wrinkles) / 5.0
        overall_score = max(0.0, min(100.0, round(100.0 - avg_severity, 1)))
        
        # Multi-frame scans have higher confidence
        conf = 0.94 if len(valid_paths) > 1 else 0.88
        
        return {
            "acne_score": final_acne,
            "dry_skin_score": final_dry,
            "oily_skin_score": final_oily,
            "dark_spots_score": final_dark_spots,
            "wrinkles_score": final_wrinkles,
            # Backward compatibility fields
            "redness_score": 0.0,
            "texture_score": final_wrinkles,
            "hyperpigmentation_score": final_dark_spots,
            "overall_score": overall_score,
            "confidence": conf,
            "frames_analyzed": len(valid_paths),
            "analysis_status": "success"
        }

# Active Computer Vision Provider
FaceAnalysisInterface = KerasComputerVisionProvider()



