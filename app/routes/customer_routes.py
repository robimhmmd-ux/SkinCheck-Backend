from fastapi import APIRouter, Depends
from app.controllers.customer_controller import CustomerController
from app.middleware.rate_limiter import rate_limit

router = APIRouter()

# Main Dashboard
router.add_api_route("/dashboard", CustomerController.dashboard, methods=["GET"], name="customer_dashboard")
router.add_api_route("/customer/dashboard", CustomerController.dashboard, methods=["GET"], name="customer_dashboard_alias")

# Profile
router.add_api_route("/profile", CustomerController.profile, methods=["GET"], name="customer_profile")
# Account deletion: Step 1 — request OTP, Step 2 — verify OTP & delete
router.add_api_route("/profile/request-delete-otp", CustomerController.request_delete_otp, methods=["POST"], name="customer_request_delete_otp")
router.add_api_route("/profile/delete", CustomerController.delete_account_action, methods=["POST"], name="customer_delete_account")

# History & Details
router.add_api_route("/history", CustomerController.history, methods=["GET"], name="customer_history")
router.add_api_route("/history/{id}", CustomerController.scan_detail, methods=["GET"], name="customer_scan_detail")
router.add_api_route("/history/{id}/retry", CustomerController.retry_analysis, methods=["POST"], name="customer_retry_analysis")

# Face Scan (multi-angle dynamic)
router.add_api_route("/scan/face", CustomerController.face_scan_view, methods=["GET"], name="customer_scan_face")
router.add_api_route("/customer/scan-face", CustomerController.face_scan_view, methods=["GET"], name="customer_scan_face_alias")
router.add_api_route("/scan-face", CustomerController.face_scan_view, methods=["GET"], name="customer_scan_face_alias2")
router.add_api_route(
    "/scan/face/upload",
    CustomerController.upload_face_scan,
    methods=["POST"],
    name="customer_scan_face_upload",
    dependencies=[Depends(rate_limit(limit=5, window=60))]
)

# AI Skincare Assistant (Ven Floating Widget)
router.add_api_route("/assistant", CustomerController.ai_assistant_view, methods=["GET"], name="customer_assistant")
router.add_api_route(
    "/assistant/history",
    CustomerController.ai_assistant_history,
    methods=["GET"],
    name="customer_assistant_history"
)
router.add_api_route(
    "/assistant/chat",
    CustomerController.ai_assistant_chat,
    methods=["POST"],
    name="customer_assistant_chat",
    dependencies=[Depends(rate_limit(limit=10, window=60))]
)
router.add_api_route(
    "/assistant/clear",
    CustomerController.ai_assistant_clear,
    methods=["POST"],
    name="customer_assistant_clear"
)

# Secure Image Serve (ownership-checked)
router.add_api_route("/storage/scans/{filename}", CustomerController.secure_image_serve, methods=["GET"], name="secure_image_serve")

