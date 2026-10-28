import logging
from sqlalchemy.orm import Session
from app.models.skin_scan import SkinScan
from app.models.analysis_result import AnalysisResult
from app.services.ai_service import FaceAnalysisInterface
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)

class SkinAnalysisService:
    """
    Business logic service for executing skin analysis workflow.
    Coordinates between Database, SkinScan, and AI Providers.
    """
    
    @staticmethod
    def run_analysis(scan_id: int, db: Session) -> bool:
        """
        Executes the AI analysis for a given single-image skin scan.
        """
        scan = db.query(SkinScan).filter(SkinScan.id == scan_id).first()
        if not scan:
            logger.error(f"Scan {scan_id} not found.")
            return False
            
        image_path = ImageService.get_image_path(scan.image_path)
        return SkinAnalysisService.run_multi_frame_analysis(scan_id, [str(image_path)], db)

    @staticmethod
    def run_multi_frame_analysis(scan_id: int, image_paths: list[str], db: Session) -> bool:
        """
        Executes the AI analysis across multiple captured frames (multi-angle dynamic scan).
        Updates the AnalysisResult and SkinScan status accordingly.
        Returns True if successful, False if failed.
        """
        scan = db.query(SkinScan).filter(SkinScan.id == scan_id).first()
        if not scan:
            logger.error(f"Scan {scan_id} not found.")
            return False
            
        analysis_record = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == scan_id).first()
        if not analysis_record:
            analysis_record = AnalysisResult(skin_scan_id=scan_id)
            db.add(analysis_record)
            
        try:
            # Update state to processing
            scan.status = "processing"
            db.commit()
            
            # Call AI Provider with multi-frame batch
            ai_result = FaceAnalysisInterface.analyze_multi_frame_skin_condition(image_paths)
            
            # Map AI result to database model (5 Facial Conditions)
            analysis_record.acne_score = ai_result.get("acne_score", 0.0)
            analysis_record.dry_skin_score = ai_result.get("dry_skin_score", 0.0)
            analysis_record.oily_skin_score = ai_result.get("oily_skin_score", 0.0)
            analysis_record.dark_spots_score = ai_result.get("dark_spots_score", 0.0)
            analysis_record.wrinkles_score = ai_result.get("wrinkles_score", 0.0)
            
            # Compatibility fields
            analysis_record.redness_score = ai_result.get("redness_score", 0.0)
            analysis_record.texture_score = ai_result.get("texture_score", analysis_record.wrinkles_score)
            analysis_record.hyperpigmentation_score = ai_result.get("hyperpigmentation_score", analysis_record.dark_spots_score)
            analysis_record.overall_score = ai_result.get("overall_score", 0.0)
            
            conf = ai_result.get('confidence', 0.92)
            conf_pct = conf * 100 if conf <= 1.0 else conf
            frames_count = ai_result.get('frames_analyzed', len(image_paths))
            
            if frames_count > 1:
                analysis_record.ai_response = (
                    f"Analisis Dinamis Selesai ({frames_count} frame multi-sudut: Kanan, Kiri, Dagu, Dahi, Jarak Dekat). "
                    f"Tingkat Keyakinan AI: {conf_pct:.0f}%."
                )
            else:
                analysis_record.ai_response = (
                    f"Analisis Selesai (1 Foto). Model: VGG16 Facial Condition. Tingkat Keyakinan: {conf_pct:.0f}%."
                )
            
            # Mark scan as completed
            scan.status = "completed"
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"AI Analysis Failed for scan {scan_id}: {str(e)}")
            scan.status = "failed"
            analysis_record.ai_response = "Gagal memproses gambar. Layanan AI tidak merespons."
            db.commit()
            return False
