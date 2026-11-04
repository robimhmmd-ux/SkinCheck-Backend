from abc import ABC, abstractmethod
import random
import time
import os
import logging
import numpy as np

# Ensure oneDNN does not conflict with TensorFlow on Windows CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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


class YuNetFaceDetector:
    """
    OpenCV YuNet ONNX Face Detector.
    Detects bounding box, confidence, and 5 landmarks (right eye, left eye, nose, mouth corners).
    """

    def __init__(self, model_path: str = "models/face_detection_yunet_2023mar.onnx"):
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            from pathlib import Path

            root = Path(__file__).resolve().parents[2]
            candidates = [
                root / "models" / "face_detection_yunet_2023mar.onnx",
                root / "facial-condition-model" / "face_detection_yunet_2023mar.onnx",
            ]
            for cand in candidates:
                if cand.exists():
                    model_path = str(cand)
                    break
        self.model_path = str(model_path)
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        try:
            import cv2

            if os.path.exists(self.model_path):
                self.detector = cv2.FaceDetectorYN.create(
                    self.model_path, "", (320, 320), score_threshold=0.6, nms_threshold=0.3
                )
                logger.info("YuNet face detector initialized successfully.")
            else:
                logger.warning(f"YuNet model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize YuNet detector: {e}")
            self.detector = None

    def detect_faces(self, img_bgr: np.ndarray, score_threshold: float = 0.6) -> list:
        """Detects faces in a BGR image. Returns list of face dictionaries."""
        if self.detector is None:
            self._init_detector()
            if self.detector is None:
                return []

        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(score_threshold)
        retval, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            conf = float(face[-1])
            if conf < score_threshold:
                continue
            x, y, fw, fh = [int(v) for v in face[:4]]
            landmarks = {
                "right_eye": (float(face[4]), float(face[5])),
                "left_eye": (float(face[6]), float(face[7])),
                "nose": (float(face[8]), float(face[9])),
                "mouth_right": (float(face[10]), float(face[11])),
                "mouth_left": (float(face[12]), float(face[13])),
            }
            results.append({
                "bbox": (x, y, fw, fh),
                "confidence": conf,
                "landmarks": landmarks,
            })
        return results

    def detect_face(self, img_input) -> bool:
        """Validates if a face is detected in the image with confidence >= 0.6."""
        import cv2

        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                return False
        else:
            img = img_input

        faces = self.detect_faces(img, score_threshold=0.6)
        return len(faces) > 0

    @staticmethod
    def crop_face(img_bgr: np.ndarray, bbox: tuple, margin: float = 0.20) -> np.ndarray:
        """Crops face region with 20% margin around bounding box."""
        h_img, w_img = img_bgr.shape[:2]
        x, y, w, h = bbox
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)
        return img_bgr[y1:y2, x1:x2]

    @staticmethod
    def estimate_pose(face: dict, frame_w: int, frame_h: int) -> str:
        """Estimates facial pose: center, left, right, up, down, close, far."""
        x, y, w, h = face["bbox"]
        bbox_area = w * h
        frame_area = max(1, frame_w * frame_h)
        ratio_area = bbox_area / frame_area

        if ratio_area > 0.45:
            return "close"
        if ratio_area < 0.10:
            return "far"

        lm = face["landmarks"]
        rx, ry = lm["right_eye"]
        lx, ly = lm["left_eye"]
        nx, ny = lm["nose"]
        mr_x, mr_y = lm["mouth_right"]
        ml_x, ml_y = lm["mouth_left"]

        # Horizontal yaw: nose position relative to eye midpoint
        eye_cx = (rx + lx) / 2.0
        eye_dist = max(1.0, abs(lx - rx))
        yaw_offset = (nx - eye_cx) / eye_dist
        if yaw_offset < -0.15:
            return "right"
        if yaw_offset > 0.15:
            return "left"

        # Pitch: nose relative to eye level and mouth level
        eye_cy = (ry + ly) / 2.0
        mouth_cy = (mr_y + ml_y) / 2.0
        face_v_dist = max(1.0, mouth_cy - eye_cy)
        pitch_ratio = (ny - eye_cy) / face_v_dist
        if pitch_ratio < 0.40:
            return "up"
        if pitch_ratio > 0.65:
            return "down"

        return "center"


class AIProviderInterface(ABC):
    """
    Abstract interface for Computer Vision AI Providers.
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
        Analyzes multiple captured frames and aggregates the scores.
        """
        pass


class MockComputerVisionProvider(AIProviderInterface):
    """
    Mock implementation of AI Provider for testing phase only.
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
            "analysis_status": "success",
        }

    def analyze_multi_frame_skin_condition(self, image_paths: list[str]) -> dict:
        result = self.analyze_skin_condition(image_paths[0] if image_paths else "")
        result["frames_analyzed"] = len(image_paths)
        result["confidence"] = 0.94
        return result


# Label pose yang dikirim frontend saat perekaman multi-pose, dipakai
# untuk pernyataan perbandingan (bukan estimasi YuNet — di backend live
# tanpa model lokal, YuNet tidak jalan, jadi label instruksi inilah yang
# menjadi pengelompoknya). Key yang tidak dikenal tampil apa adanya.
POSE_LABELS_ID = {
    "center": "Hadap Depan",
    "left": "Pipi Kiri",
    "right": "Pipi Kanan",
    "up": "Area Dagu",
    "down": "Area Dahi",
    "close": "Detail Tekstur",
    "far": "Jarak Jauh",
    "all": "Semua frame",
}

_CONDITION_LABELS_ID = [
    ("acne_score", "jerawat"),
    ("dry_skin_score", "kulit kering"),
    ("oily_skin_score", "kulit berminyak"),
    ("dark_spots_score", "flek hitam"),
    ("wrinkles_score", "kerutan"),
]


def _aggregate_modal_scores(results):
    """Agregasi skor dari beberapa frame Modal, rumusnya sama seperti
    jalur lokal: peak 35% untuk jerawat, flek hitam, kerutan, dan kulit
    berminyak; rata-rata untuk kulit kering."""
    def peak(vals):
        if len(vals) == 1:
            return float(vals[0])
        ordered = sorted(vals)
        k = max(1, int(np.ceil(len(ordered) * 0.35)))
        return float(np.mean(ordered[-k:]))

    acne = round(peak([r["acne_score"] for r in results]), 1)
    dark = round(peak([r["dark_spots_score"] for r in results]), 1)
    wrinkles = round(peak([r["wrinkles_score"] for r in results]), 1)
    dry = round(float(np.mean([r["dry_skin_score"] for r in results])), 1)
    oily = round(peak([r["oily_skin_score"] for r in results]), 1)
    avg_severity = (acne + dry + oily + dark + wrinkles) / 5.0
    return {
        "acne_score": acne,
        "dry_skin_score": dry,
        "oily_skin_score": oily,
        "dark_spots_score": dark,
        "wrinkles_score": wrinkles,
        "redness_score": 0.0,
        "texture_score": wrinkles,
        "hyperpigmentation_score": dark,
        "overall_score": max(0.0, min(100.0, round(100.0 - avg_severity, 1))),
        "confidence": round(sum(r.get("confidence", 0.92) for r in results) / len(results), 2),
    }


def _build_pose_statements(per_pose, overall_score, total_passed):
    """Pernyataan perbandingan antar pose, murni komparatif: pose mana
    yang skornya tertinggi/terendah. Bukan diagnosis wilayah presisi —
    satu pose adalah foto seluruh wajah dari sudut itu, bukan crop
    wilayah tersebut saja."""
    statements = [
        f"Rata-rata keseluruhan dari {total_passed} frame yang berhasil dianalisis: skor keseluruhan {overall_score}%."
    ]
    scored = {k: v for k, v in per_pose.items() if v.get("overall_score") is not None}
    if len(scored) < 2:
        return statements
    spreads = []
    for key, label in _CONDITION_LABELS_ID:
        vals = [(v[key], k) for k, v in scored.items() if v.get(key) is not None]
        if len(vals) < 2:
            continue
        top_val, top_pose = max(vals)
        low_val, low_pose = min(vals)
        spreads.append((top_val - low_val, label, top_pose, top_val, low_pose, low_val))
    spreads.sort(key=lambda s: s[0], reverse=True)
    for spread, label, top_pose, top_val, low_pose, low_val in spreads[:2]:
        if spread < 1:
            continue
        statements.append(
            f"Skor {label} paling tinggi terlihat pada pose "
            f"{POSE_LABELS_ID.get(top_pose, top_pose)} ({top_val}%), paling rendah "
            f"pada pose {POSE_LABELS_ID.get(low_pose, low_pose)} ({low_val}%)."
        )
    return statements


class KerasComputerVisionProvider(AIProviderInterface):
    """
    Real AI Provider that combines YuNet (Face Detection) + VGG16 (Skin Analysis).
    - YuNet: Validates face presence, bounding box, landmarks, and pose.
    - VGG16: Detects 5 skin conditions: Acne, Dry Skin, Oily Skin, Dark Spots, Wrinkles.
    - Preprocessing: 20% margin crop -> CLAHE in LAB -> resize 224x224 -> normalize / 255.0.
    """

    def __init__(self, model_path: str = "facial-condition-model/finetuned_best.h5"):
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
        self.yunet = YuNetFaceDetector()

        if os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(f"Model not found at {self.model_path}. Will attempt to load later.")

    def _load_model(self):
        try:
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf

            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info("Successfully loaded Keras VGG16 model.")
        except Exception as e:
            logger.error(f"Failed to load VGG16 model: {e}")

    def _call_modal_analyze(self, image_path: str) -> dict:
        """Forward image to serverless Modal.com AI endpoint when local model is absent."""
        import httpx
        from app.config.settings import settings

        modal_url = settings.MODAL_AI_URL.rstrip("/") + "/analyze"
        try:
            with open(str(image_path), "rb") as f:
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(modal_url, files=files)
                if resp.status_code == 422:
                    detail = resp.json().get("detail", "Tidak terdeteksi wajah pada gambar.")
                    raise ValueError(detail)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "acne_score": data["acne_score"],
                    "dry_skin_score": data["dry_skin_score"],
                    "oily_skin_score": data["oily_skin_score"],
                    "dark_spots_score": data["dark_spots_score"],
                    "wrinkles_score": data["wrinkles_score"],
                    "redness_score": 0.0,
                    "texture_score": data["wrinkles_score"],
                    "hyperpigmentation_score": data["dark_spots_score"],
                    "overall_score": data["overall_score"],
                    "confidence": data.get("confidence", 0.92),
                    "frames_analyzed": 1,
                    # Modal menganalisis 1 frame (yang dikirim) — dan di sisi
                    # Modal frame itu sudah lolos YuNet lalu diprediksi VGG.
                    # Tanpa dua key ini, halaman scan memakai default 0 dan
                    # statistiknya tampil 0/0 padahal analisisnya berhasil.
                    "frames_passed_yunet": 1,
                    "frames_used_vgg": 1,
                    "analysis_status": "success",
                }
        except httpx.HTTPError as e:
            logger.error(f"Modal AI endpoint request error: {e}")
            raise RuntimeError(f"Gagal menghubungi server AI Modal: {e}")

    def _modal_per_pose_analysis(self, image_paths: list[str], poses) -> dict:
        """Fallback Modal per pose: panggil /analyze untuk tiap frame
        (paralel, 4 worker), kelompokkan hasilnya menurut label pose yang
        dikirim frontend, agregasi per pose, lalu agregasi keseluruhan.
        Frame tanpa wajah (422 dari Modal) dilewati; kalau Modal tidak
        bisa dihubungi sama sekali, errornya diteruskan apa adanya supaya
        scan gagal dengan jujur, bukan dengan skor sebagian yang diam-diam
        tidak lengkap. Jalur lokal tidak memakai ini — di sana pose
        diestimasi YuNet dan inferensi jalan batch."""
        from concurrent.futures import ThreadPoolExecutor

        if poses is not None and len(poses) == len(image_paths):
            grouped = {}
            for path, pose in zip(image_paths, poses):
                grouped.setdefault(pose or "all", []).append(path)
        else:
            grouped = {"all": list(image_paths)}

        def _run(item):
            pose_key, path = item
            try:
                return pose_key, self._call_modal_analyze(path), None
            except Exception as e:  # noqa: BLE001 - diklasifikasi pemanggil
                return pose_key, None, e

        items = [(k, p) for k, paths in grouped.items() for p in paths]
        with ThreadPoolExecutor(max_workers=4) as ex:
            outcomes = list(ex.map(_run, items))

        grouped_results = {k: [] for k in grouped}
        first_runtime_error = None
        first_unexpected_error = None
        for pose_key, result, err in outcomes:
            if result is not None:
                grouped_results[pose_key].append(result)
            elif isinstance(err, RuntimeError):
                if first_runtime_error is None:
                    first_runtime_error = err
            elif not isinstance(err, ValueError):
                if first_unexpected_error is None:
                    first_unexpected_error = err
            # ValueError = frame tanpa wajah (422): frame dilewati saja.

        per_pose = {}
        all_results = []
        for pose_key, paths in grouped.items():
            results = grouped_results[pose_key]
            if results:
                entry = _aggregate_modal_scores(results)
                entry["frames_captured"] = len(paths)
                entry["frames_passed"] = len(results)
                per_pose[pose_key] = entry
                all_results.extend(results)
            else:
                per_pose[pose_key] = {
                    "acne_score": None, "dry_skin_score": None, "oily_skin_score": None,
                    "dark_spots_score": None, "wrinkles_score": None, "redness_score": None,
                    "texture_score": None, "hyperpigmentation_score": None, "overall_score": None,
                    "confidence": None, "frames_captured": len(paths), "frames_passed": 0,
                }

        if not all_results:
            if first_runtime_error is not None:
                raise first_runtime_error
            if first_unexpected_error is not None:
                raise first_unexpected_error
            raise ValueError("Tidak terdeteksi wajah pada frame yang direkam. Ulangi scan dengan wajah menghadap kamera.")

        overall = _aggregate_modal_scores(all_results)
        overall.update({
            "frames_analyzed": len(all_results),
            "frames_captured": len(image_paths),
            "frames_passed_yunet": len(all_results),
            "frames_used_vgg": len(all_results),
            "per_pose": per_pose,
            "statements": _build_pose_statements(per_pose, overall["overall_score"], len(all_results)),
            "analysis_status": "success",
        })
        return overall

    def detect_face(self, image_path: str) -> bool:
        """Validates face presence using real YuNet face detector."""
        return self.yunet.detect_face(image_path)

    def analyze_skin_condition(self, image_path: str) -> dict:
        """
        Analyzes a single image:
        1. Runs YuNet face detection gate.
        2. Crops face bounding box with 20% margin.
        3. Preprocesses with CLAHE in LAB and normalizes to (224, 224, 3).
        4. Predicts 5 skin condition scores using VGG16.
        """
        if not self.model:
            self._load_model()
            if not self.model:
                logger.info("Local VGG16 model not loaded; delegating to Modal.com AI...")
                return self._call_modal_analyze(image_path)

        import cv2

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError("File gambar tidak dapat dibaca.")

        # 1. Real face gate via YuNet
        faces = self.yunet.detect_faces(img_bgr, score_threshold=0.6)
        if not faces:
            raise ValueError("Tidak terdeteksi wajah pada gambar. Pastikan wajah terlihat jelas dan pencahayaan cukup.")

        # Pick highest confidence face
        primary_face = max(faces, key=lambda f: f["confidence"])

        # 2. Crop face with 20% margin
        cropped_bgr = self.yunet.crop_face(img_bgr, primary_face["bbox"], margin=0.20)
        if cropped_bgr.size == 0:
            cropped_bgr = img_bgr

        # 3. CLAHE enhancement and normalization
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = enhance_skin_image(cropped_rgb, is_rgb=True)
        resized = cv2.resize(enhanced_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        norm_img = resized.astype(np.float32) / 255.0
        batch_tensor = np.expand_dims(norm_img, axis=0)

        # 4. VGG16 prediction
        predictions = self.model.predict(batch_tensor, verbose=0)[0]

        # Sigmoid outputs: [0: Acne, 1: Dry Skin, 2: Oily Skin, 3: Dark Spots, 4: Wrinkles]
        acne_score = round(float(predictions[0] * 100.0), 1)
        dry_score = round(float(predictions[1] * 100.0), 1)
        oily_score = round(float(predictions[2] * 100.0), 1)
        dark_spots_score = round(float(predictions[3] * 100.0), 1)
        wrinkles_score = round(float(predictions[4] * 100.0), 1)

        avg_severity = (acne_score + dry_score + oily_score + dark_spots_score + wrinkles_score) / 5.0
        overall_score = max(0.0, min(100.0, round(100.0 - avg_severity, 1)))

        return {
            "acne_score": acne_score,
            "dry_skin_score": dry_score,
            "oily_skin_score": oily_score,
            "dark_spots_score": dark_spots_score,
            "wrinkles_score": wrinkles_score,
            "redness_score": 0.0,
            "texture_score": wrinkles_score,
            "hyperpigmentation_score": dark_spots_score,
            "overall_score": overall_score,
            "confidence": round(primary_face["confidence"], 2),
            "frames_analyzed": 1,
            "analysis_status": "success",
        }

    def analyze_multi_frame_skin_condition(self, image_paths: list[str], poses=None) -> dict:
        """
        Analyzes multi-frame capture (e.g. 60-second webcam stream):
        1. Runs YuNet on each frame, filters out non-face and tiny faces.
        2. Groups passed frames by estimated pose (center, left, right, up, down, close, far).
        3. Samples top 1-2 highest confidence frames per pose (12-20 frames total).
        4. Crops, applies CLAHE, normalizes, and runs single-pass batch inference on VGG16.
        5. Aggregates scores (peak 35% for localized blemishes, mean for dry skin).

        `poses` adalah label pose per frame dari frontend (sejajar dengan
        image_paths). Label ini hanya dipakai jalur delegasi Modal di
        bawah; jalur lokal tetap memakai estimasi pose YuNet.
        """
        if not image_paths:
            raise ValueError("Daftar frame gambar tidak boleh kosong.")

        if not self.model:
            self._load_model()
            if not self.model:
                logger.info("Local VGG16 model not loaded; delegating to Modal.com AI per pose...")
                return self._modal_per_pose_analysis(image_paths, poses)

        import cv2

        total_captured = len(image_paths)
        valid_frames_by_pose = {
            "center": [],
            "left": [],
            "right": [],
            "up": [],
            "down": [],
            "close": [],
            "far": [],
        }
        total_passed_yunet = 0

        # Step 1: Filter and group with YuNet
        for path in image_paths:
            try:
                img_bgr = cv2.imread(str(path))
                if img_bgr is None:
                    continue
                h_f, w_f = img_bgr.shape[:2]
                faces = self.yunet.detect_faces(img_bgr, score_threshold=0.6)
                if not faces:
                    continue

                best_face = max(faces, key=lambda f: f["confidence"])
                fw, fh = best_face["bbox"][2], best_face["bbox"][3]
                if fw < 50 or fh < 50:
                    continue  # Face too small/distant

                total_passed_yunet += 1
                pose = self.yunet.estimate_pose(best_face, w_f, h_f)
                valid_frames_by_pose[pose].append({
                    "path": path,
                    "img_bgr": img_bgr,
                    "face": best_face,
                })
            except Exception as frame_err:
                logger.warning(f"Error filtering frame {path}: {frame_err}")

        if total_passed_yunet == 0:
            raise ValueError("Tidak terdeteksi wajah pada frame yang direkam. Ulangi scan dengan wajah menghadap kamera.")

        # Step 2: Sample top 1-3 frames per pose
        sampled_items = []
        for pose_name, items in valid_frames_by_pose.items():
            if not items:
                continue
            # Sort by confidence descending
            sorted_items = sorted(items, key=lambda x: x["face"]["confidence"], reverse=True)
            k = 3 if pose_name == "center" else 2
            sampled_items.extend(sorted_items[:k])

        # If sampled items is empty (fallback to all passed)
        if not sampled_items:
            for pose_name, items in valid_frames_by_pose.items():
                sampled_items.extend(items[:2])

        # Step 3: Crop and preprocess sampled frames
        batch_arrays = []
        for item in sampled_items:
            img_bgr = item["img_bgr"]
            face = item["face"]
            cropped = self.yunet.crop_face(img_bgr, face["bbox"], margin=0.20)
            if cropped.size == 0:
                cropped = img_bgr

            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            enhanced = enhance_skin_image(rgb, is_rgb=True)
            resized = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            batch_arrays.append(norm)

        # Step 4: Batch VGG16 Inference
        batch_tensor = np.array(batch_arrays, dtype=np.float32)
        predictions = self.model.predict(batch_tensor, verbose=0)

        acne_scores = predictions[:, 0] * 100.0
        dry_scores = predictions[:, 1] * 100.0
        oily_scores = predictions[:, 2] * 100.0
        dark_spots_scores = predictions[:, 3] * 100.0
        wrinkles_scores = predictions[:, 4] * 100.0

        # Step 5: Multi-Angle Aggregation
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

        return {
            "acne_score": final_acne,
            "dry_skin_score": final_dry,
            "oily_skin_score": final_oily,
            "dark_spots_score": final_dark_spots,
            "wrinkles_score": final_wrinkles,
            "redness_score": 0.0,
            "texture_score": final_wrinkles,
            "hyperpigmentation_score": final_dark_spots,
            "overall_score": overall_score,
            "confidence": 0.94 if len(sampled_items) > 1 else 0.88,
            "frames_analyzed": len(sampled_items),
            "frames_captured": total_captured,
            "frames_passed_yunet": total_passed_yunet,
            "frames_used_vgg": len(sampled_items),
            "analysis_status": "success",
        }


# Active Computer Vision Provider
FaceAnalysisInterface = KerasComputerVisionProvider()
