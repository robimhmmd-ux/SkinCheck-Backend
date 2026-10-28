from fastapi import Request, Depends, Form, Query
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from math import ceil

from app.config import settings
from app.config.database import get_db
from app.middleware.auth_guard import require_admin
from app.models.product import Product

templates = Jinja2Templates(directory=str(settings.TEMPLATE_DIR))

class AdminController:
    """Controller handling Internal Web Admin Dashboard and operations."""

    @staticmethod
    async def dashboard(request: Request, user: dict = Depends(require_admin), db: Session = Depends(get_db)):
        """Render Admin Dashboard with dynamic metrics."""
        from datetime import datetime, date
        from sqlalchemy import func
        from app.models.user import User
        from app.models.skin_scan import SkinScan
        from app.models.product import Product
        from app.models.analysis_result import AnalysisResult
        from app.models.chat_message import ChatMessage

        today = date.today()

        total_customers = db.query(User).filter(User.role == "customer").count()
        active_customers = db.query(User).filter(User.role == "customer", User.status == "active").count()
        
        total_scans = db.query(SkinScan).count()
        scans_today = db.query(SkinScan).filter(func.date(SkinScan.created_at) == today).count()
        
        total_products = db.query(Product).filter(Product.status == "active").count()

        success_analysis = db.query(SkinScan).filter(SkinScan.status == "completed").count()
        failed_analysis = db.query(SkinScan).filter(SkinScan.status == "failed").count()
        
        total_analysis = success_analysis + failed_analysis
        success_rate = f"{(success_analysis / total_analysis * 100):.1f}%" if total_analysis > 0 else "0%"
        total_ai_sessions = db.query(ChatMessage.user_id).distinct().count()

        metrics = {
            "total_customers": total_customers,
            "active_users": active_customers,
            "total_scans": total_scans,
            "scans_today": scans_today,
            "total_products": total_products,
            "success_rate": success_rate,
            "failed_analysis": failed_analysis,
            "total_ai_sessions": total_ai_sessions
        }

        context = {
            "request": request,
            "page_title": "Dashboard",
            "app_name": settings.APP_NAME,
            "metrics": metrics
        }
        return templates.TemplateResponse("pages/admin/dashboard.html", context)

    @staticmethod
    async def users_index(
        request: Request, 
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        search: str = Query(None),
        status: str = Query(None),
        page: int = Query(1, ge=1)
    ):
        """List users with search, filter, and pagination."""
        from math import ceil
        from app.models.user import User
        limit = 10
        offset = (page - 1) * limit

        query = db.query(User).filter(User.role == "customer")
        if search:
            query = query.filter(User.name.ilike(f"%{search}%") | User.email.ilike(f"%{search}%"))
        if status:
            query = query.filter(User.status == status)
            
        total = query.count()
        users = query.order_by(User.id.desc()).offset(offset).limit(limit).all()
        total_pages = ceil(total / limit) if total > 0 else 1

        # Calculate total scans for each user
        from sqlalchemy import func
        from app.models.skin_scan import SkinScan
        user_scans = dict(db.query(SkinScan.user_id, func.count(SkinScan.id)).group_by(SkinScan.user_id).all())

        context = {
            "request": request,
            "page_title": "User Management",
            "app_name": settings.APP_NAME,
            "users": users,
            "user_scans": user_scans,
            "search": search or "",
            "status": status or "",
            "page": page,
            "total_pages": total_pages,
            "total": total
        }
        return templates.TemplateResponse("pages/admin/users/index.html", context)

    @staticmethod
    async def user_create_view(request: Request, user: dict = Depends(require_admin)):
        context = {
            "request": request,
            "page_title": "Add User",
            "app_name": settings.APP_NAME,
            "user_obj": None
        }
        return templates.TemplateResponse("pages/admin/users/form.html", context)

    @staticmethod
    async def user_store(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        name: str = Form(...),
        email: str = Form(...),
        password: str = Form(...),
        status: str = Form("active")
    ):
        from app.models.user import User
        from app.utils.auth import get_password_hash
        
        # Check if email exists
        if db.query(User).filter(User.email == email).first():
            return RedirectResponse(url="/admin/users/create?error=Email already exists", status_code=303)
            
        new_user = User(
            name=name,
            email=email,
            password_hash=get_password_hash(password),
            role="customer",
            status=status
        )
        db.add(new_user)
        db.commit()
        return RedirectResponse(url="/admin/users", status_code=303)

    @staticmethod
    async def user_edit_view(
        request: Request, 
        user_id: int, 
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        from app.models.user import User
        user_obj = db.query(User).filter(User.id == user_id, User.role == "customer").first()
        if not user_obj:
            return RedirectResponse(url="/admin/users", status_code=303)
            
        context = {
            "request": request,
            "page_title": f"Edit User #{user_obj.id}",
            "app_name": settings.APP_NAME,
            "user_obj": user_obj
        }
        return templates.TemplateResponse("pages/admin/users/form.html", context)

    @staticmethod
    async def user_update(
        request: Request,
        user_id: int,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        name: str = Form(...),
        email: str = Form(...),
        password: str = Form(""),
        status: str = Form("active")
    ):
        from app.models.user import User
        from app.utils.auth import get_password_hash
        user_obj = db.query(User).filter(User.id == user_id, User.role == "customer").first()
        if not user_obj:
            return RedirectResponse(url="/admin/users", status_code=303)
            
        # Check if email exists for another user
        existing = db.query(User).filter(User.email == email, User.id != user_id).first()
        if existing:
            return RedirectResponse(url=f"/admin/users/{user_id}/edit?error=Email already exists", status_code=303)
            
        user_obj.name = name
        user_obj.email = email
        user_obj.status = status
        
        if password.strip():
            user_obj.password_hash = get_password_hash(password)
            
        db.commit()
        return RedirectResponse(url="/admin/users", status_code=303)

    @staticmethod
    async def user_delete(
        request: Request,
        user_id: int,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        from app.models.user import User
        user_obj = db.query(User).filter(User.id == user_id, User.role == "customer").first()
        if user_obj:
            user_obj.status = "inactive"
            db.commit()
        return RedirectResponse(url="/admin/users", status_code=303)

    @staticmethod
    async def scans_index(
        request: Request, 
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        page: int = Query(1, ge=1)
    ):
        """List Skin Scans with pagination."""
        from math import ceil
        from sqlalchemy.orm import aliased
        from app.models.skin_scan import SkinScan
        from app.models.user import User
        from app.models.analysis_result import AnalysisResult
        
        limit = 10
        offset = (page - 1) * limit

        query = db.query(SkinScan, User, AnalysisResult).join(
            User, SkinScan.user_id == User.id
        ).outerjoin(
            AnalysisResult, SkinScan.id == AnalysisResult.skin_scan_id
        )
        
        total = query.count()
        results = query.order_by(SkinScan.id.desc()).offset(offset).limit(limit).all()
        total_pages = ceil(total / limit) if total > 0 else 1

        context = {
            "request": request,
            "page_title": "Skin Scans Monitoring",
            "app_name": settings.APP_NAME,
            "results": results,
            "page": page,
            "total_pages": total_pages,
            "total": total
        }
        return templates.TemplateResponse("pages/admin/scans/index.html", context)

    @staticmethod
    async def logs_index(
        request: Request, 
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        page: int = Query(1, ge=1)
    ):
        """List Activity Logs with pagination."""
        from math import ceil
        from app.models.activity_log import ActivityLog
        from app.models.user import User
        
        limit = 20
        offset = (page - 1) * limit

        query = db.query(ActivityLog, User).join(
            User, ActivityLog.user_id == User.id
        )
        
        total = query.count()
        results = query.order_by(ActivityLog.id.desc()).offset(offset).limit(limit).all()
        total_pages = ceil(total / limit) if total > 0 else 1

        context = {
            "request": request,
            "page_title": "Activity Logs",
            "app_name": settings.APP_NAME,
            "results": results,
            "page": page,
            "total_pages": total_pages,
            "total": total
        }
        return templates.TemplateResponse("pages/admin/logs/index.html", context)

    @staticmethod
    async def products_index(
        request: Request, 
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        search: str = Query(None),
        category: str = Query(None),
        page: int = Query(1, ge=1)
    ):
        """List products with search, filter, and pagination."""
        limit = 10
        offset = (page - 1) * limit

        query = db.query(Product)
        if search:
            query = query.filter(Product.name.ilike(f"%{search}%") | Product.brand.ilike(f"%{search}%"))
        if category:
            query = query.filter(Product.category == category)
            
        total = query.count()
        products = query.order_by(Product.id.desc()).offset(offset).limit(limit).all()
        
        total_pages = ceil(total / limit) if total > 0 else 1

        context = {
            "request": request,
            "page_title": "Product Management",
            "app_name": settings.APP_NAME,
            "products": products,
            "search": search or "",
            "category": category or "",
            "page": page,
            "total_pages": total_pages,
            "total": total
        }
        return templates.TemplateResponse("pages/admin/products/index.html", context)

    @staticmethod
    async def product_create_view(request: Request, user: dict = Depends(require_admin)):
        """View form to create a new product."""
        context = {
            "request": request,
            "page_title": "Add Product",
            "app_name": settings.APP_NAME,
            "product": None
        }
        return templates.TemplateResponse("pages/admin/products/form.html", context)

    @staticmethod
    async def product_store(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        brand: str = Form(...),
        name: str = Form(...),
        category: str = Form(...),
        description: str = Form(None),
        usage_guidance: str = Form(None),
        ingredients: str = Form(None),
        skin_concern: str = Form(None),
        shopee_link: str = Form(None),
        tokopedia_link: str = Form(None),
        status: str = Form("active")
    ):
        """Save a new product."""
        product = Product(
            brand=brand,
            name=name,
            category=category,
            description=description,
            usage_guidance=usage_guidance,
            shopee_link=shopee_link or None,
            tokopedia_link=tokopedia_link or None,
            status=status
        )
        if ingredients:
            product.set_ingredients([i.strip() for i in ingredients.split(",") if i.strip()])
        if skin_concern:
            product.set_skin_concern([s.strip() for s in skin_concern.split(",") if s.strip()])
            
        db.add(product)
        db.commit()
        
        return RedirectResponse(url="/admin/products", status_code=303)

    @staticmethod
    async def product_edit_view(
        request: Request, 
        product_id: int, 
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """View form to edit a product."""
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return RedirectResponse(url="/admin/products", status_code=303)
            
        context = {
            "request": request,
            "page_title": f"Edit Product #{product.id}",
            "app_name": settings.APP_NAME,
            "product": product,
            "ingredients_str": ", ".join(product.get_ingredients()),
            "concerns_str": ", ".join(product.get_skin_concern())
        }
        return templates.TemplateResponse("pages/admin/products/form.html", context)

    @staticmethod
    async def product_update(
        request: Request,
        product_id: int,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        brand: str = Form(...),
        name: str = Form(...),
        category: str = Form(...),
        description: str = Form(None),
        usage_guidance: str = Form(None),
        ingredients: str = Form(None),
        skin_concern: str = Form(None),
        shopee_link: str = Form(None),
        tokopedia_link: str = Form(None),
        status: str = Form("active")
    ):
        """Update an existing product."""
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return RedirectResponse(url="/admin/products", status_code=303)
            
        product.brand = brand
        product.name = name
        product.category = category
        product.description = description
        product.usage_guidance = usage_guidance
        product.shopee_link = shopee_link or None
        product.tokopedia_link = tokopedia_link or None
        product.status = status
        
        if ingredients:
            product.set_ingredients([i.strip() for i in ingredients.split(",") if i.strip()])
        else:
            product.set_ingredients([])
            
        if skin_concern:
            product.set_skin_concern([s.strip() for s in skin_concern.split(",") if s.strip()])
        else:
            product.set_skin_concern([])
            
        db.commit()
        return RedirectResponse(url="/admin/products", status_code=303)

    @staticmethod
    async def product_delete(
        request: Request,
        product_id: int,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Soft delete (deactivate) a product."""
        product = db.query(Product).filter(Product.id == product_id).first()
        if product:
            product.status = "inactive"
            db.commit()
            
        return RedirectResponse(url="/admin/products", status_code=303)

    # ==========================================
    # AI COPILOT & PRODUCT PROPOSAL
    # ==========================================
    @staticmethod
    async def ai_copilot_view(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Render AI Copilot Trend Analytics & Skincare Product Proposals."""
        from app.services.ai_copilot_service import AICopilotService
        insights = AICopilotService.generate_ai_copilot_insights(db)
        
        context = {
            "request": request,
            "page_title": "AI Copilot — Analisis Tren Kulit",
            "app_name": settings.APP_NAME,
            "insights": insights
        }
        return templates.TemplateResponse("pages/admin/ai_copilot.html", context)

    @staticmethod
    async def product_auto_insert(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        brand: str = Form(...),
        name: str = Form(...),
        category: str = Form(...),
        description: str = Form(None),
        usage_guidance: str = Form(None),
        ingredients: str = Form(None),
        skin_concern: str = Form(None),
        shopee_link: str = Form(None),
        tokopedia_link: str = Form(None)
    ):
        """Auto-inserts the approved AI Copilot skincare proposal into the product catalog."""
        from app.services.ai_copilot_service import AICopilotService
        admin_data = request.session.get("admin") or {}
        admin_id = admin_data.get("id")

        proposal_data = {
            "brand": brand,
            "name": name,
            "category": category,
            "description": description,
            "usage_guidance": usage_guidance,
            "ingredients": ingredients,
            "skin_concern": skin_concern,
            "shopee_link": shopee_link,
            "tokopedia_link": tokopedia_link
        }

        product = AICopilotService.auto_insert_proposed_product(db, proposal_data, admin_user_id=admin_id)
        return RedirectResponse(
            url=f"/admin/products",
            status_code=303
        )

    @staticmethod
    async def sync_rag_knowledge(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Endpoint to refresh product catalog context in RAG vector/memory store."""
        from app.services.ai_copilot_service import AICopilotService
        from fastapi.responses import JSONResponse
        res = AICopilotService.sync_rag_knowledge(db)
        return JSONResponse(content=res)

    # ==========================================
    # PENGATURAN SISTEM & API KEY
    # ==========================================
    # PENGATURAN SISTEM, BATASAN AI, & PENGUJI KONEKSI EKSTERNAL
    # ==========================================
    @staticmethod
    async def settings_view(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        success: str = None
    ):
        """Render System Configuration, AI Boundaries, Flow Control & External Connection Testing."""
        from app.models.system_setting import SystemSetting
        
        # Load AI Settings
        gemini_api_key = SystemSetting.get_value(db, "GEMINI_API_KEY", getattr(settings, "GEMINI_API_KEY", ""))
        masked_gemini_key = (gemini_api_key[:6] + "..." + gemini_api_key[-4:]) if len(gemini_api_key) > 10 else ("Tersimpan" if gemini_api_key else "Belum disetel")
        ai_model = SystemSetting.get_value(db, "AI_MODEL", "gemini-1.5-flash")
        ai_max_tokens = SystemSetting.get_value(db, "AI_MAX_TOKENS", "1024")
        ai_temperature = SystemSetting.get_value(db, "AI_TEMPERATURE", "0.7")
        ai_daily_scan_limit = SystemSetting.get_value(db, "AI_DAILY_SCAN_LIMIT", "5")
        ai_simulation_mode = SystemSetting.get_value(db, "AI_SIMULATION_MODE", "0")

        # Load Flow & Registration Settings
        registration_otp_mode = SystemSetting.get_value(db, "REGISTRATION_OTP_MODE", "dual")
        maintenance_mode = SystemSetting.get_value(db, "MAINTENANCE_MODE", "off")

        # Load Email & WA Settings
        mail_username = SystemSetting.get_value(db, "MAIL_USERNAME", getattr(settings, "MAIL_USERNAME", ""))
        mail_from_name = SystemSetting.get_value(db, "MAIL_FROM_NAME", getattr(settings, "MAIL_FROM_NAME", "SkinCheck AI"))
        wa_api_token = SystemSetting.get_value(db, "WA_API_TOKEN", getattr(settings, "WA_API_TOKEN", ""))
        wa_api_url = SystemSetting.get_value(db, "WA_API_URL", getattr(settings, "WA_API_URL", "https://api.fonnte.com/send"))

        # Vision Parameters
        vgg_threshold = SystemSetting.get_value(db, "VGG_THRESHOLD", "40")
        yolo_confidence = SystemSetting.get_value(db, "YOLO_CONFIDENCE", "0.25")

        admin_session = request.session.get("admin") or {}

        context = {
            "request": request,
            "page_title": "Pengaturan Sistem, Batasan AI & Penguji Koneksi",
            "app_name": settings.APP_NAME,
            "success": success,
            "admin_email": admin_session.get("email", "admin@skincheck.internal"),
            "settings_data": {
                "gemini_api_key": gemini_api_key,
                "masked_gemini_key": masked_gemini_key,
                "ai_model": ai_model,
                "ai_max_tokens": ai_max_tokens,
                "ai_temperature": ai_temperature,
                "ai_daily_scan_limit": ai_daily_scan_limit,
                "ai_simulation_mode": ai_simulation_mode,
                "registration_otp_mode": registration_otp_mode,
                "maintenance_mode": maintenance_mode,
                "mail_username": mail_username,
                "mail_from_name": mail_from_name,
                "wa_api_token": wa_api_token,
                "wa_api_url": wa_api_url,
                "vgg_threshold": vgg_threshold,
                "yolo_confidence": yolo_confidence
            }
        }
        return templates.TemplateResponse("pages/admin/settings.html", context)

    @staticmethod
    async def settings_update(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        gemini_api_key: str = Form(None),
        ai_model: str = Form("gemini-1.5-flash"),
        ai_max_tokens: str = Form("1024"),
        ai_temperature: str = Form("0.7"),
        ai_daily_scan_limit: str = Form("5"),
        ai_simulation_mode: str = Form("0"),
        registration_otp_mode: str = Form("dual"),
        maintenance_mode: str = Form("off"),
        mail_username: str = Form(None),
        mail_from_name: str = Form("SkinCheck AI"),
        wa_api_token: str = Form(None),
        wa_api_url: str = Form("https://api.fonnte.com/send"),
        vgg_threshold: str = Form("40"),
        yolo_confidence: str = Form("0.25")
    ):
        """Update system settings, AI limits, and flow controls in MySQL database."""
        from app.models.system_setting import SystemSetting
        from app.models.activity_log import ActivityLog
        
        # Save AI parameters & limits
        if gemini_api_key is not None and not gemini_api_key.startswith("..."):
            SystemSetting.set_value(db, "GEMINI_API_KEY", gemini_api_key.strip(), "Gemini LLM API Key")
        SystemSetting.set_value(db, "AI_MODEL", ai_model.strip(), "Model LLM Aktif")
        SystemSetting.set_value(db, "AI_MAX_TOKENS", ai_max_tokens.strip(), "Batasan Maksimal Output Token AI")
        SystemSetting.set_value(db, "AI_TEMPERATURE", ai_temperature.strip(), "Tingkat Kreativitas AI")
        SystemSetting.set_value(db, "AI_DAILY_SCAN_LIMIT", ai_daily_scan_limit.strip(), "Batas Kuota Scan Wajah Harian")
        SystemSetting.set_value(db, "AI_SIMULATION_MODE", ai_simulation_mode.strip(), "Mode Simulasi AI Fallback")

        # Save Flow Controls
        SystemSetting.set_value(db, "REGISTRATION_OTP_MODE", registration_otp_mode.strip(), "Mode Alur Verifikasi OTP Registrasi")
        SystemSetting.set_value(db, "MAINTENANCE_MODE", maintenance_mode.strip(), "Status Mode Pemeliharaan Sistem")

        # Save Gateways
        if mail_username is not None:
            SystemSetting.set_value(db, "MAIL_USERNAME", mail_username.strip(), "SMTP Email Account")
        if mail_from_name is not None:
            SystemSetting.set_value(db, "MAIL_FROM_NAME", mail_from_name.strip(), "SMTP Sender Display Name")
        if wa_api_token is not None:
            SystemSetting.set_value(db, "WA_API_TOKEN", wa_api_token.strip(), "WhatsApp Gateway Token")
        if wa_api_url is not None:
            SystemSetting.set_value(db, "WA_API_URL", wa_api_url.strip(), "WhatsApp Gateway API Endpoint")
        if vgg_threshold is not None:
            SystemSetting.set_value(db, "VGG_THRESHOLD", vgg_threshold.strip(), "VGG16 Clinical Threshold %")
        if yolo_confidence is not None:
            SystemSetting.set_value(db, "YOLO_CONFIDENCE", yolo_confidence.strip(), "YOLOv8 Acne Confidence Threshold")

        # Log Activity
        admin_data = request.session.get("admin") or {}
        if admin_data.get("id"):
            log = ActivityLog(
                user_id=admin_data.get("id"),
                activity=f"Admin memperbarui konfigurasi sistem, batasan AI (Model {ai_model}, limit {ai_daily_scan_limit}/hari), dan flow registrasi ({registration_otp_mode})",
                ip_address="127.0.0.1"
            )
            db.add(log)
            db.commit()

        return RedirectResponse(url="/admin/settings?success=Pengaturan+sistem,+batasan+AI,+dan+alur+proyek+berhasil+disimpan!", status_code=303)

    # ──────────────────────────────────────────
    # PENGUJI SISTEM KONEKSI EKSTERNAL (TESTING HUB)
    # ──────────────────────────────────────────
    @staticmethod
    async def test_email_connection(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Tester: Pengujian koneksi ke layanan eksternal SMTP Gmail."""
        import smtplib
        import time
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from app.models.system_setting import SystemSetting
        from app.models.activity_log import ActivityLog
        from fastapi.responses import JSONResponse

        data = await request.form()
        target_email = data.get("test_email") or "admin@skincheck.internal"
        mail_user = SystemSetting.get_value(db, "MAIL_USERNAME", getattr(settings, "MAIL_USERNAME", ""))
        mail_pass = SystemSetting.get_value(db, "MAIL_PASSWORD", getattr(settings, "MAIL_PASSWORD", ""))
        mail_from = SystemSetting.get_value(db, "MAIL_FROM_NAME", getattr(settings, "MAIL_FROM_NAME", "SkinCheck AI"))

        if not mail_user or not mail_pass:
            return JSONResponse({
                "success": False,
                "message": "Akun email SMTP (MAIL_USERNAME/MAIL_PASSWORD) belum disetel di pengaturan."
            })

        start = time.time()
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "[SkinCheck AI] Pengujian Sistem Koneksi OTP Email"
            msg["From"] = f"{mail_from} <{mail_user}>"
            msg["To"] = target_email
            html_body = f"""
            <div style="font-family:sans-serif;padding:20px;border:1px solid #38bdf8;border-radius:10px;background:#f0f9ff;">
                <h3 style="color:#0284c7;">Uji Koneksi Layanan Eksternal OTP Email</h3>
                <p>Halo Admin, ini adalah pesan uji koneksi otomatis dari SkinCheck System Diagnostic.</p>
                <p><b>Status:</b> Terhubung &amp; Berfungsi Normal</p>
                <p><b>Waktu Uji:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr style="border:none;border-top:1px solid #bae6fd;">
                <small style="color:#64748b;">SkinCheck AI &bull; Automated System Verification</small>
            </div>
            """
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP("smtp.gmail.com", 587, timeout=8) as server:
                server.ehlo()
                server.starttls()
                server.login(mail_user, mail_pass)
                server.sendmail(mail_user, target_email, msg.as_string())

            latency = int((time.time() - start) * 1000)

            # Log test
            admin_data = request.session.get("admin") or {}
            if admin_data.get("id"):
                db.add(ActivityLog(
                    user_id=admin_data.get("id"),
                    activity=f"Admin menguji koneksi eksternal OTP Email ke {target_email}: Berhasil ({latency} ms)",
                    ip_address="127.0.0.1"
                ))
                db.commit()

            return JSONResponse({
                "success": True,
                "latency_ms": latency,
                "message": f"Koneksi SMTP berhasil! Email uji coba telah terkirim ke {target_email} ({latency} ms)."
            })
        except smtplib.SMTPAuthenticationError:
            return JSONResponse({
                "success": False,
                "message": "Autentikasi SMTP Gmail Gagal: Gunakan Google App Password 16-karakter (bukan sandi login biasa)."
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "message": f"Gagal menghubungi server SMTP: {str(e)}"
            })

    @staticmethod
    async def test_wa_connection(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Tester: Pengujian koneksi ke layanan eksternal WhatsApp Gateway."""
        import json
        import urllib.request
        import time
        from app.models.system_setting import SystemSetting
        from app.models.activity_log import ActivityLog
        from fastapi.responses import JSONResponse

        data = await request.form()
        target_phone = data.get("test_phone") or "081234567890"
        wa_token = SystemSetting.get_value(db, "WA_API_TOKEN", getattr(settings, "WA_API_TOKEN", ""))
        wa_url = SystemSetting.get_value(db, "WA_API_URL", getattr(settings, "WA_API_URL", "https://api.fonnte.com/send"))

        if not wa_token:
            return JSONResponse({
                "success": True,
                "mode": "Simulation / Dev Mode",
                "message": f"[Dev Mode] Gateway WhatsApp siap. Token belum disetel, pesan uji coba disimulasikan ke log terminal untuk nomor {target_phone}."
            })

        start = time.time()
        try:
            payload = json.dumps({
                "target": target_phone,
                "message": f"[SkinCheck AI] Uji Koneksi Gateway WhatsApp Berhasil pada {time.strftime('%Y-%m-%d %H:%M:%S')}."
            }).encode("utf-8")

            req = urllib.request.Request(
                wa_url,
                data=payload,
                headers={"Authorization": wa_token, "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                status_code = response.status
                body = response.read().decode("utf-8")

            latency = int((time.time() - start) * 1000)

            # Log
            admin_data = request.session.get("admin") or {}
            if admin_data.get("id"):
                db.add(ActivityLog(
                    user_id=admin_data.get("id"),
                    activity=f"Admin menguji koneksi eksternal WhatsApp Gateway: Status {status_code} ({latency} ms)",
                    ip_address="127.0.0.1"
                ))
                db.commit()

            return JSONResponse({
                "success": status_code in (200, 201),
                "latency_ms": latency,
                "response": body[:200],
                "message": f"Koneksi WhatsApp Gateway terverifikasi dengan status {status_code} ({latency} ms)!"
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "message": f"Koneksi ke gateway WhatsApp gagal: {str(e)}"
            })

    @staticmethod
    async def test_ai_connection(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Tester: Pengujian koneksi ke layanan AI eksternal (Google Gemini API)."""
        import time
        from app.models.system_setting import SystemSetting
        from app.models.activity_log import ActivityLog
        from fastapi.responses import JSONResponse

        api_key = SystemSetting.get_value(db, "GEMINI_API_KEY", getattr(settings, "GEMINI_API_KEY", ""))
        ai_model = SystemSetting.get_value(db, "AI_MODEL", "gemini-1.5-flash")

        if not api_key:
            return JSONResponse({
                "success": False,
                "message": "Google Gemini API Key belum disetel di sistem."
            })

        start = time.time()
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(ai_model)
            response = model.generate_content("Katakan 'Koneksi AI SkinCheck Berhasil' dan beri 1 tips singkat perawatan kulit.")
            latency = int((time.time() - start) * 1000)

            admin_data = request.session.get("admin") or {}
            if admin_data.get("id"):
                db.add(ActivityLog(
                    user_id=admin_data.get("id"),
                    activity=f"Admin menguji koneksi Google Gemini AI ({ai_model}): Berhasil ({latency} ms)",
                    ip_address="127.0.0.1"
                ))
                db.commit()

            return JSONResponse({
                "success": True,
                "model": ai_model,
                "latency_ms": latency,
                "sample_output": response.text.strip(),
                "message": f"Koneksi AI Google Gemini ({ai_model}) aktif & responsif ({latency} ms)!"
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "message": f"Gagal menghubungi Google Gemini API: {str(e)}"
            })

    @staticmethod
    async def test_db_connection(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db)
    ):
        """Tester: Pengujian koneksi ke database internal XAMPP MySQL."""
        import time
        from sqlalchemy import text
        from fastapi.responses import JSONResponse

        start = time.time()
        try:
            res = db.execute(text("SHOW TABLES;")).fetchall()
            tables = [r[0] for r in res]
            latency = round((time.time() - start) * 1000, 2)

            return JSONResponse({
                "success": True,
                "database": getattr(settings, "DB_DATABASE", "skincheck_db"),
                "connection": getattr(settings, "DB_CONNECTION", "mysql"),
                "tables_count": len(tables),
                "tables": tables,
                "latency_ms": latency,
                "message": f"Koneksi Database XAMPP MySQL ('skincheck_db') sangat cepat ({latency} ms) dengan {len(tables)} tabel siap pakai!"
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "message": f"Gagal menghubungi XAMPP MySQL: {str(e)}"
            })

    # ==========================================
    # PROFIL ADMIN
    # ==========================================
    @staticmethod
    async def profile_view(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        success: str = None,
        error: str = None
    ):
        """Render Admin Profile & Security Settings."""
        from app.models.user import User
        admin_session = request.session.get("admin") or {}
        admin_user = db.query(User).filter(User.id == admin_session.get("id")).first()
        if not admin_user:
            admin_user = db.query(User).filter(User.role == "admin").first()

        context = {
            "request": request,
            "page_title": "Profil Admin",
            "app_name": settings.APP_NAME,
            "admin": admin_user,
            "success": success,
            "error": error
        }
        return templates.TemplateResponse("pages/admin/profile.html", context)

    @staticmethod
    async def profile_update(
        request: Request,
        user: dict = Depends(require_admin),
        db: Session = Depends(get_db),
        name: str = Form(...),
        email: str = Form(...),
        old_password: str = Form(None),
        new_password: str = Form(None)
    ):
        """Update admin profile details and password."""
        from app.models.user import User
        from app.utils.auth import verify_password, get_password_hash
        from app.models.activity_log import ActivityLog

        admin_session = request.session.get("admin") or {}
        admin_user = db.query(User).filter(User.id == admin_session.get("id")).first()
        if not admin_user:
            admin_user = db.query(User).filter(User.role == "admin").first()

        admin_user.name = name.strip()
        admin_user.email = email.strip().lower()

        if new_password:
            if not old_password or not verify_password(old_password, admin_user.password_hash):
                return templates.TemplateResponse("pages/admin/profile.html", {
                    "request": request,
                    "page_title": "Profil Admin",
                    "app_name": settings.APP_NAME,
                    "admin": admin_user,
                    "error": "Password lama tidak sesuai."
                })
            if len(new_password) < 6:
                return templates.TemplateResponse("pages/admin/profile.html", {
                    "request": request,
                    "page_title": "Profil Admin",
                    "app_name": settings.APP_NAME,
                    "admin": admin_user,
                    "error": "Password baru minimal 6 karakter."
                })
            admin_user.password_hash = get_password_hash(new_password)

        db.commit()

        # Update session
        request.session["admin"] = {
            "id": admin_user.id,
            "name": admin_user.name,
            "email": admin_user.email,
            "role": admin_user.role
        }

        # Log
        if admin_user and admin_user.id:
            log = ActivityLog(
                user_id=admin_user.id,
                activity=f"Admin {admin_user.name} memperbarui profil dan kredensial",
                ip_address="127.0.0.1"
            )
            db.add(log)
            db.commit()

        return RedirectResponse(url="/admin/profile?success=Profil+admin+berhasil+diperbarui!", status_code=303)
