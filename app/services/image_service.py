import os
import uuid
from pathlib import Path
from fastapi import UploadFile
from PIL import Image, ImageStat
import io

# Define secure storage path outside of the static folder.
# On Vercel the repo filesystem is read-only except /tmp, so uploads
# go to /tmp there (ephemeral per instance). Locally, or when
# SKINCHECK_STORAGE_DIR is set, that directory is used instead.
if os.environ.get("SKINCHECK_STORAGE_DIR"):
    STORAGE_DIR = Path(os.environ["SKINCHECK_STORAGE_DIR"])
elif os.environ.get("VERCEL"):
    STORAGE_DIR = Path("/tmp/storage/scans")
else:
    STORAGE_DIR = Path("storage/scans")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

class ImageValidationException(Exception):
    """Exception raised for failed image validation."""
    pass

class ImageService:
    """Service to handle secure image upload, validation, and storage."""
    
    @staticmethod
    async def process_and_save_scan(file: UploadFile) -> str:
        """
        Validates the uploaded image and saves it securely.
        Returns the filename if successful.
        """
        # 1. Read file into memory
        file_bytes = await file.read()
        
        # 2. Check File Size (e.g. limit to 5MB)
        if len(file_bytes) > 5 * 1024 * 1024:
            raise ImageValidationException("Ukuran file terlalu besar (Maksimal 5MB).")
            
        try:
            # 3. Validate it is an actual image using Pillow (ignores client MIME type)
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # verify checks if it's a valid image without decoding completely
            
            # Re-open for actual processing since verify() closes the file pointer
            image = Image.open(io.BytesIO(file_bytes))
            
            # 4. Check Minimum Resolution (e.g., 300x300)
            width, height = image.size
            if width < 300 or height < 300:
                raise ImageValidationException("Resolusi gambar terlalu rendah (Minimal 300x300 piksel).")
                
            # 5. Check if image is too dark (Average brightness)
            # Convert to grayscale to evaluate brightness
            grayscale_image = image.convert('L')
            stat = ImageStat.Stat(grayscale_image)
            avg_brightness = stat.mean[0]
            
            if avg_brightness < 40: # Arbitrary threshold for "too dark"
                raise ImageValidationException("Kondisi pencahayaan terlalu gelap. Harap cari tempat yang lebih terang.")
                
            # 6. Save image securely
            # Generate random UUID for filename to prevent enumeration and path traversal
            file_extension = "png" # We'll enforce saving as PNG for scans
            filename = f"{uuid.uuid4().hex}.{file_extension}"
            filepath = STORAGE_DIR / filename
            
            # Save the image (converts to RGB if it was RGBA to avoid issues, though PNG supports RGBA)
            image.save(filepath, format="PNG")
            
            return filename
            
        except ImageValidationException as e:
            raise e
        except Exception as e:
            raise ImageValidationException(f"Gagal memproses gambar: File tidak valid atau rusak. ({str(e)})")

    @staticmethod
    def get_image_path(filename: str) -> Path:
        """Returns the absolute path for a given stored scan filename."""
        filepath = STORAGE_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError("Gambar tidak ditemukan.")
        return filepath
