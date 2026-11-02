"""Modal.com Deployment for SkinCheck AI Models (YuNet + VGG16)

Hosts the TensorFlow VGG16 model (174 MB) and OpenCV YuNet ONNX model (232 KB)
serverless on Modal.com to provide 24/7 high-performance REST API endpoints.
"""

import os
import modal

app = modal.App("skincheck-ai")

# Container image definition with Python 3.11, TensorFlow, OpenCV, Pillow, FastAPI
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libsm6", "libxext6")
    .run_commands(
        "pip install --upgrade pip",
        "pip install tensorflow==2.17.0 opencv-python-headless==4.10.0.84 'numpy<2.0.0' pillow fastapi python-multipart",
    )
    .add_local_file(
        "models/face_detection_yunet_2023mar.onnx",
        remote_path="/root/models/face_detection_yunet_2023mar.onnx",
    )
    .add_local_file(
        "facial-condition-model/finetuned_best.h5",
        remote_path="/root/models/finetuned_best.h5",
    )
)


@app.cls(
    image=image,
    cpu=2.0,
    memory=4096,
    scaledown_window=300,
)
class SkinCheckAI:
    @modal.enter()
    def load_models(self):
        import cv2
        import tensorflow as tf

        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        self.yunet_path = "/root/models/face_detection_yunet_2023mar.onnx"
        self.vgg_path = "/root/models/finetuned_best.h5"

        # Initialize YuNet Face Detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Initialize VGG16 Skin Condition Scorer
        self.model = tf.keras.models.load_model(self.vgg_path, compile=False)

    def _enhance_clahe(self, img_rgb):
        import cv2

        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return cv2.addWeighted(enhanced_rgb, 0.75, img_rgb, 0.25, 0)

    def _detect_and_crop(self, img_bgr):
        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None, None

        best = max(faces, key=lambda f: f[14])
        if best[14] < 0.6:
            return None, None

        box = best[0:4].astype(int)
        x, y, bw, bh = box
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        cropped = img_bgr[y1:y2, x1:x2]
        return (cropped if cropped.size > 0 else img_bgr), float(best[14])

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import cv2
        import numpy as np

        web_app = FastAPI(
            title="SkinCheck AI Engine",
            description="Production Face Detection (YuNet) and Facial Condition Analysis (VGG16) hosted on Modal.com",
            version="2.0.0",
        )

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @web_app.get("/")
        @web_app.get("/health")
        def health():
            return {
                "status": "online",
                "service": "SkinCheck Modal AI Engine",
                "models": ["YuNet 2023mar ONNX", "VGG16 Finetuned H5"],
            }

        @web_app.post("/analyze")
        async def analyze(file: UploadFile = File(...)):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return JSONResponse(status_code=400, content={"detail": "Format gambar tidak valid."})

            cropped, conf = self._detect_and_crop(img_bgr)
            if cropped is None:
                return JSONResponse(
                    status_code=422,
                    content={
                        "status": "failed",
                        "detail": "Tidak terdeteksi wajah pada gambar. Pastikan wajah terlihat jelas dan tegak ke kamera.",
                    },
                )

            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            enhanced = self._enhance_clahe(rgb)
            resized = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            batch = np.expand_dims(norm, axis=0)

            preds = self.model.predict(batch, verbose=0)[0]
            acne = round(float(preds[0]) * 100.0, 1)
            dry = round(float(preds[1]) * 100.0, 1)
            oily = round(float(preds[2]) * 100.0, 1)
            dark_spots = round(float(preds[3]) * 100.0, 1)
            wrinkles = round(float(preds[4]) * 100.0, 1)

            avg_sev = (acne + dry + oily + dark_spots + wrinkles) / 5.0
            overall = max(0.0, min(100.0, round(100.0 - avg_sev, 1)))

            return {
                "status": "success",
                "confidence": conf,
                "acne_score": acne,
                "dry_skin_score": dry,
                "oily_skin_score": oily,
                "dark_spots_score": dark_spots,
                "wrinkles_score": wrinkles,
                "overall_score": overall,
            }

        return web_app
