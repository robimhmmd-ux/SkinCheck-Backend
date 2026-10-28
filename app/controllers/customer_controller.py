import logging
from fastapi import Request, Depends, HTTPException, UploadFile, Query
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.config import settings
from app.config.database import get_db
from app.middleware.auth_guard import require_customer
from app.models.user import User
from app.models.skin_scan import SkinScan
from app.models.analysis_result import AnalysisResult

templates = Jinja2Templates(directory=str(settings.TEMPLATE_DIR))
logger = logging.getLogger(__name__)

class CustomerController:
    """Controller handling Customer-facing Dashboard and interactions."""

    @staticmethod
    async def dashboard(
        request: Request, 
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Render Customer Dashboard with real data from database."""
        db_user = db.query(User).filter(User.id == user["id"]).first()
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        profile_data = db_user.get_profile_data()

        user_info = {
            "name": db_user.name,
            "email": db_user.email,
            "joined_date": db_user.created_at.strftime("%d %B %Y"),
            "skin_goals": profile_data.get("skin_concerns", [])
        }

        # Get latest scan
        latest_scan = db.query(SkinScan).filter(SkinScan.user_id == user["id"]).order_by(desc(SkinScan.created_at)).first()
        
        scan_data = None
        if latest_scan:
            analysis = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == latest_scan.id).first()
            
            scan_data = {
                "id": latest_scan.id,
                "scan_id": latest_scan.id,
                "timestamp": latest_scan.created_at.strftime("%d %B %Y, %H:%M WIB"),
                "overall_score": int(analysis.overall_score) if analysis and analysis.overall_score is not None else 0,
                "acne_score": int(analysis.acne_score) if analysis and analysis.acne_score is not None else 0,
                "dry_skin_score": int(analysis.dry_skin_score) if analysis and analysis.dry_skin_score is not None else 0,
                "oily_skin_score": int(analysis.oily_skin_score) if analysis and analysis.oily_skin_score is not None else 0,
                "dark_spots_score": int(analysis.dark_spots_score) if analysis and analysis.dark_spots_score is not None else (int(analysis.hyperpigmentation_score) if analysis and analysis.hyperpigmentation_score is not None else 0),
                "wrinkles_score": int(analysis.wrinkles_score) if analysis and analysis.wrinkles_score is not None else (int(analysis.texture_score) if analysis and analysis.texture_score is not None else 0),
            }

        # Get scan history (limit to 5 for dashboard)
        history_records = db.query(SkinScan).filter(SkinScan.user_id == user["id"]).order_by(desc(SkinScan.created_at)).limit(5).all()
        scan_history = []
        for record in history_records:
            h_analysis = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == record.id).first()
            scan_history.append({
                "id": record.id,
                "date": record.created_at.strftime("%d %b %Y"),
                "overall": h_analysis.overall_score if h_analysis else "-",
                "status": record.status.capitalize()
            })

        # Calculate skin condition averages and historical trends across user scans
        from app.models.product import Product
        all_user_scans = (
            db.query(SkinScan)
            .filter(SkinScan.user_id == user["id"])
            .order_by(SkinScan.created_at.asc())
            .all()
        )

        trend_labels = []
        trend_acne = []
        trend_oily = []
        trend_dry = []
        trend_dark_spots = []
        trend_wrinkles = []
        trend_overall = []

        sum_acne = 0
        sum_oily = 0
        sum_dry = 0
        sum_dark_spots = 0
        sum_wrinkles = 0
        sum_overall = 0
        analyzed_count = 0

        for sc in all_user_scans:
            ar = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == sc.id).first()
            if ar and ar.overall_score is not None:
                acne = int(ar.acne_score or 0)
                oily = int(ar.oily_skin_score or 0)
                dry = int(ar.dry_skin_score or 0)
                spots = int(ar.dark_spots_score if ar.dark_spots_score is not None else (ar.hyperpigmentation_score or 0))
                wrinkles = int(ar.wrinkles_score if ar.wrinkles_score is not None else (ar.texture_score or 0))
                overall = int(ar.overall_score or 0)

                date_label = sc.created_at.strftime("%d %b") if sc.created_at else f"#{sc.id}"
                trend_labels.append(date_label)
                trend_acne.append(acne)
                trend_oily.append(oily)
                trend_dry.append(dry)
                trend_dark_spots.append(spots)
                trend_wrinkles.append(wrinkles)
                trend_overall.append(overall)

                sum_acne += acne
                sum_oily += oily
                sum_dry += dry
                sum_dark_spots += spots
                sum_wrinkles += wrinkles
                sum_overall += overall
                analyzed_count += 1

        if analyzed_count > 0:
            skin_averages = {
                "count": analyzed_count,
                "acne": round(sum_acne / analyzed_count, 1),
                "oily": round(sum_oily / analyzed_count, 1),
                "dry": round(sum_dry / analyzed_count, 1),
                "dark_spots": round(sum_dark_spots / analyzed_count, 1),
                "wrinkles": round(sum_wrinkles / analyzed_count, 1),
                "overall": round(sum_overall / analyzed_count, 1),
            }
        else:
            skin_averages = {
                "count": 0,
                "acne": 0,
                "oily": 0,
                "dry": 0,
                "dark_spots": 0,
                "wrinkles": 0,
                "overall": 0,
            }

        trend_data = {
            "labels": trend_labels,
            "acne": trend_acne,
            "oily": trend_oily,
            "dry": trend_dry,
            "dark_spots": trend_dark_spots,
            "wrinkles": trend_wrinkles,
            "overall": trend_overall,
            "has_data": analyzed_count > 0
        }

        total_scans_count = len(all_user_scans)

        # Evaluasi kecenderungan profil kulit berdasarkan akumulasi data scan
        skin_tendency = {
            "type": "Belum Teridentifikasi",
            "badge_color": "secondary",
            "summary": "Belum ada rekaman pemindaian kulit. Silakan lakukan scan wajah pertama Anda untuk mendeteksi kecenderungan kondisi kulit.",
            "dominant_issue": "Data Belum Tersedia",
            "recommendation": "Gunakan pembersih wajah lembut dan gunakan sunscreen di pagi hari.",
            "actives": ["Niacinamide", "Ceramide", "Hyaluronic Acid"]
        }

        ref_overall = scan_data["overall_score"] if scan_data else (skin_averages["overall"] if skin_averages["count"] > 0 else 0)
        if ref_overall > 0 or skin_averages["count"] > 0:
            acne_val = scan_data["acne_score"] if scan_data else skin_averages["acne"]
            oily_val = scan_data["oily_skin_score"] if scan_data else skin_averages["oily"]
            dry_val = scan_data["dry_skin_score"] if scan_data else skin_averages["dry"]
            spots_val = scan_data["dark_spots_score"] if scan_data else skin_averages["dark_spots"]
            wrinkles_val = scan_data["wrinkles_score"] if scan_data else skin_averages["wrinkles"]

            if acne_val >= 45 and oily_val >= 45:
                skin_tendency = {
                    "type": "Cenderung Berminyak & Rawan Jerawat (Oily-Acne Prone)",
                    "badge_color": "danger",
                    "summary": "Aktivitas kelenjar sebum di area T-Zone dan pipi relatif tinggi disertai potensi peradangan pori. Kulit membutuhkan regulasi minyak dan agen anti-inflamasi non-komedogenik.",
                    "dominant_issue": "Produksi Sebum Berlebih & Inflamasi Pori",
                    "recommendation": "Terapkan double cleansing di malam hari, toner dengan Salicylic Acid (BHA 1-2%), dan pelembap water-based gel.",
                    "actives": ["Salicylic Acid (BHA)", "Niacinamide 2-5%", "Tea Tree Extract", "Zinc PCA"]
                }
            elif oily_val >= 50 and dry_val >= 40:
                skin_tendency = {
                    "type": "Kombinasi Dehidrasi (Dehydrated Oily Skin)",
                    "badge_color": "warning",
                    "summary": "Kulit memproduksi sebum berlebih sebagai kompensasi atas defisit kelembapan alami skin barrier. Permukaan terasa berminyak namun lapisan dalam mengalami dehidrasi.",
                    "dominant_issue": "Disregulasi Barrier & Sebum",
                    "recommendation": "Fokus pada hidrasi berlapis dengan humektan (Hyaluronic Acid/Glycerin) dan kunci dengan ceramide ringan tanpa menyumbat pori.",
                    "actives": ["Hyaluronic Acid", "Ceramide Complex", "Centella Asiatica", "Panthenol (B5)"]
                }
            elif oily_val >= 50:
                skin_tendency = {
                    "type": "Cenderung Berminyak (Oily Complexion)",
                    "badge_color": "teal",
                    "summary": "Sekresi minyak alami tinggi sehingga area wajah tampak berkilau. Pori-pori cenderung membesar namun elastisitas jaringan kulit terhidrasi baik.",
                    "dominant_issue": "Kelebihan Minyak & Kilap T-Zone",
                    "recommendation": "Gunakan pembersih wajah gel berbusa lembut, toner bebas alkohol, dan pelembap mattifying bertekstur gel.",
                    "actives": ["Niacinamide", "Zinc PCA", "Green Tea Extract", "Clay Mask Mingguan"]
                }
            elif dry_val >= 45:
                skin_tendency = {
                    "type": "Cenderung Kering & Sensitif (Dry-Sensitive Barrier)",
                    "badge_color": "info",
                    "summary": "Kadar lipid dan hidrasi stratum korneum rendah. Kulit rentan terasa tertarik setelah cuci muka, bersisik halus, dan sensitif terhadap perubahan cuaca.",
                    "dominant_issue": "Defisit Hidrasi & Kelemahan Skin Barrier",
                    "recommendation": "Gunakan gentle cream cleanser non-SLS, pelembap kaya ceramide & asam lemak esensial, serta hindari eksfoliasi fisik yang kasar.",
                    "actives": ["Ceramide NP/AP/EOP", "Squalane", "Glycerin", "Colloidal Oatmeal"]
                }
            elif spots_val >= 45:
                skin_tendency = {
                    "type": "Cenderung Hiperpigmentasi & Noda Hitam (Pigmented-Uneven)",
                    "badge_color": "primary",
                    "summary": "Distribusi pigmen melanin meningkat, tampak dari bekas noda jerawat (PIH) atau bercak paparan sinar UV matahari yang membutuhkan proteksi dan pencerah aktif.",
                    "dominant_issue": "Hiperpigmentasi Melanin & Warna Kulit Tidak Merata",
                    "recommendation": "Disiplin gunakan sunscreen SPF 50 PA++++ setiap pagi, dipadukan serum pencerah stabil yang menghambat tirosinase.",
                    "actives": ["Alpha Arbutin", "Vitamin C (Ethyl Ascorbic)", "Tranexamic Acid", "Licorice Extract"]
                }
            elif wrinkles_val >= 45:
                skin_tendency = {
                    "type": "Cenderung Menurun Elastisitas / Garis Halus (Early Aging)",
                    "badge_color": "accent",
                    "summary": "Kepadatan serat kolagen dan kelembapan struktural menunjukkan tanda garis ekspresi di sekitar mata dan senyum yang memerlukan stimulasi kolagen.",
                    "dominant_issue": "Elastisitas Jaringan & Garis Ekspresi",
                    "recommendation": "Gunakan serum peptida, pelembap kaya antioksidan, dan retinol/bakuchiol secara bertahap pada malam hari.",
                    "actives": ["Peptide Complex", "Retinol / Bakuchiol", "Antioxidant CoQ10", "Adenosine"]
                }
            else:
                skin_tendency = {
                    "type": "Kondisi Seimbang & Terawat (Optimal Balanced)",
                    "badge_color": "success",
                    "summary": "Indeks kesehatan kulit berada pada kisaran prima. Produksi sebum seimbang, kadar hidrasi terjaga baik, dan tekstur kulit halus dengan inflamasi minimal.",
                    "dominant_issue": "Pemeliharaan Rutinitas Harian",
                    "recommendation": "Pertahankan kebiasaan baik: Pembersih lembut, hidrasi harian, dan sunscreen SPF 30+ setiap beraktivitas di luar ruangan.",
                    "actives": ["Hyaluronic Acid", "Niacinamide 2%", "Ceramides", "Broad-Spectrum SPF"]
                }

        # Query active skincare products for customer search and discovery
        catalog_products = (
            db.query(Product)
            .filter(Product.status == "active")
            .order_by(Product.name.asc())
            .all()
        )
        products = []
        for p in catalog_products:
            products.append({
                "id": p.id,
                "brand": p.brand,
                "name": p.name,
                "category": p.category,
                "description": p.description or "Produk perawatan kulit teruji klinis dan berizin resmi BPOM.",
                "ingredients": p.get_ingredients(),
                "skin_concern": p.get_skin_concern(),
                "usage_guidance": p.usage_guidance or "",
                "shopee_link": p.shopee_link or "",
                "tokopedia_link": p.tokopedia_link or ""
            })

        import json
        context = {
            "request": request,
            "page_title": "Customer Dashboard — SkinCheck",
            "app_name": settings.APP_NAME,
            "user": user_info,
            "latest_scan": scan_data,
            "scan_history": scan_history,
            "skin_averages": skin_averages,
            "trend_data": trend_data,
            "trend_json": json.dumps(trend_data),
            "averages_json": json.dumps(skin_averages),
            "products": products,
            "total_scans_count": total_scans_count,
            "skin_tendency": skin_tendency
        }
        return templates.TemplateResponse("pages/customer/dashboard.html", context)

    @staticmethod
    async def profile(
        request: Request, 
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Render Customer Profile."""
        db_user = db.query(User).filter(User.id == user["id"]).first()
        profile_data = db_user.get_profile_data() if db_user else {}
        
        context = {
            "request": request,
            "page_title": "Profil Saya — SkinCheck",
            "app_name": settings.APP_NAME,
            "user": db_user,
            "user_detail": db_user,
            "profile_data": profile_data
        }
        return templates.TemplateResponse("pages/customer/profile.html", context)

    @staticmethod
    async def history(
        request: Request, 
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db),
        page: int = Query(1, ge=1)
    ):
        """Render all Scan History with pagination."""
        from math import ceil
        limit = 10
        offset = (page - 1) * limit
        
        query = db.query(SkinScan).filter(SkinScan.user_id == user["id"])
        total = query.count()
        scans = query.order_by(desc(SkinScan.created_at)).offset(offset).limit(limit).all()
        total_pages = ceil(total / limit) if total > 0 else 1
        
        history_list = []
        for s in scans:
            analysis = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == s.id).first()
            history_list.append({
                "scan": s,
                "analysis": analysis,
                "sunscreen": None
            })
            
        context = {
            "request": request,
            "page_title": "Riwayat Scan — SkinCheck",
            "app_name": settings.APP_NAME,
            "history_list": history_list,
            "page": page,
            "total_pages": total_pages,
            "total": total
        }
        return templates.TemplateResponse("pages/customer/history.html", context)

    @staticmethod
    async def request_delete_otp(
        request: Request,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """
        STEP 1 of account deletion: send OTP to user's email.
        Called when customer clicks 'Hapus Akun' on the profile page.
        """
        from fastapi.responses import JSONResponse
        from app.services.otp_service import send_delete_account_otp

        db_user = db.query(User).filter(User.id == user["id"]).first()
        if not db_user:
            return JSONResponse(status_code=404, content={"error": "User tidak ditemukan."})

        result = send_delete_account_otp(db_user.email, db_user.name, db)

        if result["dev_code"]:
            return {"status": "ok", "message": "OTP digenerate (mode dev).", "dev_code": result["dev_code"]}
        return {"status": "ok", "message": f"Kode konfirmasi telah dikirim ke {db_user.email}."}

    @staticmethod
    async def delete_account_action(
        request: Request,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """
        STEP 2 of account deletion: verify OTP then permanently delete account.
        Expects JSON body: { "otp_code": "123456" }
        """
        from fastapi.responses import JSONResponse, RedirectResponse
        from pathlib import Path
        from app.services.otp_service import verify_otp

        try:
            body = await request.json()
            otp_code = (body.get("otp_code") or "").strip()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Request tidak valid."})

        db_user = db.query(User).filter(User.id == user["id"]).first()
        if not db_user:
            return JSONResponse(status_code=404, content={"error": "User tidak ditemukan."})

        if not verify_otp(db_user.email, otp_code, "delete_account", db):
            return JSONResponse(status_code=400, content={"error": "Kode OTP salah atau sudah kedaluwarsa."})

        user_id = db_user.id

        # Delete physical scan images
        scans = db.query(SkinScan).filter(SkinScan.user_id == user_id).all()
        storage_dir = Path("storage/scans")
        for scan in scans:
            if scan.image_path:
                filepath = storage_dir / scan.image_path
                if filepath.exists():
                    try:
                        filepath.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete image {filepath}: {e}")

        # Permanently delete user (cascade handles related records)
        db.delete(db_user)
        db.commit()

        # Clear session
        request.session.clear()

        return {"status": "ok", "message": "Akun dan seluruh data Anda telah dihapus secara permanen."}

    @staticmethod
    async def scan_detail(
        request: Request, 
        id: int,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Render detailed view of a specific scan."""
        scan = db.query(SkinScan).filter(SkinScan.id == id, SkinScan.user_id == user["id"]).first()
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
            
        analysis = db.query(AnalysisResult).filter(AnalysisResult.skin_scan_id == scan.id).first()
        
        context = {
            "request": request,
            "page_title": f"Detail Scan #{scan.id} — SkinCheck",
            "app_name": settings.APP_NAME,
            "scan": scan,
            "analysis": analysis,
            "sunscreen": None
        }
        return templates.TemplateResponse("pages/customer/scan_detail.html", context)

    @staticmethod
    async def face_scan_view(request: Request, user: dict = Depends(require_customer)):
        """Render UI for Face Scan using WebRTC."""
        context = {
            "request": request,
            "page_title": "Mulai Face Scan — SkinCheck",
            "app_name": settings.APP_NAME,
            "scan_type": "face"
        }
        return templates.TemplateResponse("pages/customer/scan_face.html", context)
        
    @staticmethod
    async def upload_face_scan(
        request: Request,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Handle upload of captured face scan (single frame or multi-angle dynamic frames)."""
        from fastapi.responses import JSONResponse
        from app.services.image_service import ImageService, ImageValidationException
        from app.services.ai_service import FaceAnalysisInterface
        from app.services.skin_analysis_service import SkinAnalysisService
        
        try:
            form = await request.form()
            uploaded_files = form.getlist("files")
            if not uploaded_files:
                single_file = form.get("file")
                if single_file:
                    uploaded_files = [single_file]
                    
            if not uploaded_files:
                return JSONResponse(status_code=400, content={"error": "Tidak ada file gambar yang diunggah."})
                
            # Process and save all frames
            saved_filenames = []
            saved_paths = []
            
            for uf in uploaded_files:
                fname = await ImageService.process_and_save_scan(uf)
                fpath = ImageService.get_image_path(fname)
                saved_filenames.append(fname)
                saved_paths.append(str(fpath))
                
            primary_filename = saved_filenames[0]
            primary_filepath = saved_paths[0]
            
            # Face Detection check on primary frame
            if not FaceAnalysisInterface.detect_face(primary_filepath):
                return JSONResponse(status_code=400, content={"error": "Wajah tidak terdeteksi. Silakan ambil ulang foto."})
                
            # Save Scan to database
            new_scan = SkinScan(
                user_id=user["id"],
                image_path=primary_filename,
                status="pending",
                image_quality_score=0.95 if len(saved_paths) > 1 else 0.9
            )
            db.add(new_scan)
            db.commit()
            db.refresh(new_scan)
            
            # Create placeholder Analysis Result with 5 conditions
            analysis = AnalysisResult(
                skin_scan_id=new_scan.id,
                acne_score=0,
                dry_skin_score=0,
                oily_skin_score=0,
                dark_spots_score=0,
                wrinkles_score=0,
                redness_score=0,
                texture_score=0,
                hyperpigmentation_score=0,
                overall_score=0,
                ai_response="Sedang dianalisis..."
            )
            db.add(analysis)
            db.commit()
            
            # Trigger Multi-Frame AI Analysis
            SkinAnalysisService.run_multi_frame_analysis(new_scan.id, saved_paths, db)
            
            msg = f"Pemindaian {len(saved_paths)} frame berhasil dianalisis!" if len(saved_paths) > 1 else "Gambar berhasil diunggah"
            return {"status": "success", "message": msg, "scan_id": new_scan.id, "frames_count": len(saved_paths)}
            
        except ImageValidationException as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            logger.error(f"Error in upload_face_scan: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Terjadi kesalahan internal server."})

    @staticmethod
    async def retry_analysis(
        request: Request,
        id: int,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Retry a failed AI analysis."""
        from fastapi.responses import RedirectResponse
        from app.services.skin_analysis_service import SkinAnalysisService
        
        # Verify ownership
        scan = db.query(SkinScan).filter(SkinScan.id == id, SkinScan.user_id == user["id"]).first()
        if not scan:
            raise HTTPException(status_code=404, detail="Scan tidak ditemukan.")
            
        if scan.status != "failed":
            raise HTTPException(status_code=400, detail="Hanya dapat mengulang scan yang berstatus gagal.")
            
        # Run Analysis
        SkinAnalysisService.run_analysis(scan.id, db)
        
        return RedirectResponse(url=f"/history/{id}", status_code=303)

    @staticmethod
    async def secure_image_serve(
        filename: str,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Serve scan images securely, ensuring only the owner can view them."""
        from fastapi.responses import FileResponse
        from app.services.image_service import ImageService
        
        # Check ownership
        scan = db.query(SkinScan).filter(SkinScan.image_path == filename).first()
        if not scan:
            raise HTTPException(status_code=404, detail="Gambar tidak ditemukan.")
            
        if scan.user_id != user["id"]:
            # Intentionally return 404 instead of 403 to prevent enumeration
            raise HTTPException(status_code=404, detail="Gambar tidak ditemukan.")
            
        try:
            filepath = ImageService.get_image_path(filename)
            return FileResponse(path=filepath, media_type="image/png")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Gambar tidak ditemukan di penyimpanan.")

    @staticmethod
    async def ai_assistant_view(
        request: Request
    ):
        """Redirect legacy /assistant route to dashboard with open_ven parameter."""
        from fastapi.responses import RedirectResponse
        scan_id = request.query_params.get("scan_id")
        context_type = request.query_params.get("context")

        user = request.session.get("user")
        target_path = "/dashboard" if user else "/"
        redirect_url = f"{target_path}?open_ven=1"
        if scan_id:
            redirect_url += f"&scan_id={scan_id}"
        if context_type:
            redirect_url += f"&context={context_type}"
        return RedirectResponse(url=redirect_url, status_code=303)

    @staticmethod
    async def ai_assistant_history(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """API endpoint to get chat history for the logged-in customer as JSON."""
        from fastapi.responses import JSONResponse
        from app.models.chat_message import ChatMessage

        user = request.session.get("user") or request.session.get("admin")
        if not user:
            return JSONResponse(status_code=200, content={"status": "success", "history": [], "authenticated": False})

        messages = db.query(ChatMessage).filter(
            ChatMessage.user_id == user["id"]
        ).order_by(ChatMessage.created_at.asc()).all()

        history = [
            {
                "role": m.role,
                "content": m.content,
                "time": m.created_at.strftime("%H:%M") if m.created_at else ""
            }
            for m in messages
        ]
        return JSONResponse(status_code=200, content={"status": "success", "history": history, "authenticated": True})

    @staticmethod
    async def ai_assistant_clear(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """API endpoint to clear chat history for the logged-in customer."""
        from fastapi.responses import JSONResponse
        from app.models.chat_message import ChatMessage

        user = request.session.get("user") or request.session.get("admin")
        if not user:
            return JSONResponse(status_code=401, content={"error": "Silakan masuk ke akun Anda terlebih dahulu."})

        try:
            db.query(ChatMessage).filter(ChatMessage.user_id == user["id"]).delete()
            db.commit()
            return JSONResponse(status_code=200, content={"status": "success", "message": "Riwayat percakapan berhasil dibersihkan."})
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
            db.rollback()
            return JSONResponse(status_code=500, content={"error": "Gagal menghapus riwayat obrolan."})

    @staticmethod
    async def ai_assistant_chat(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """API endpoint for AI chat — receives JSON { message, scan_id? }."""
        from fastapi.responses import JSONResponse
        from app.services.ai_assistant_service import assistant

        user = request.session.get("user") or request.session.get("admin")
        if not user:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Silakan masuk ke akun Anda terlebih dahulu untuk berdiskusi dengan Ven.",
                    "require_auth": True
                }
            )

        try:
            body = await request.json()
            message = body.get("message", "").strip()
            scan_id = body.get("scan_id")

            if not message:
                return JSONResponse(status_code=400, content={"error": "Pesan tidak boleh kosong."})

            if len(message) > 2000:
                return JSONResponse(status_code=400, content={"error": "Pesan terlalu panjang (maks 2000 karakter)."})

            scan_id_int = int(scan_id) if scan_id else None
            user_role = user.get("role", "customer")

            response_text = assistant.chat(
                user_id=user["id"],
                message=message,
                db=db,
                scan_id=scan_id_int,
                role=user_role
            )

            return JSONResponse(status_code=200, content={"status": "success", "response": response_text})

        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return JSONResponse(status_code=500, content={"error": "Terjadi kesalahan pada layanan AI."})

    @staticmethod
    async def submit_feedback(
        request: Request,
        user: dict = Depends(require_customer),
        db: Session = Depends(get_db)
    ):
        """Record customer feedback or bug report to activity_logs."""
        from fastapi.responses import JSONResponse
        from app.models.activity_log import ActivityLog
        try:
            form = await request.form()
            category = form.get("category", "General Feedback")
            message = form.get("message", "").strip()
            if not message:
                return JSONResponse(status_code=400, content={"error": "Pesan laporan atau feedback tidak boleh kosong."})
            log_entry = ActivityLog(
                user_id=user["id"],
                activity=f"Feedback [{category}]: {message[:220]}",
                ip_address=request.client.host if request.client else None
            )
            db.add(log_entry)
            db.commit()
            return JSONResponse(status_code=200, content={"status": "success", "message": "Terima kasih! Feedback & laporan kendala Anda berhasil terkirim."})
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Terjadi kesalahan saat menyimpan feedback."})

