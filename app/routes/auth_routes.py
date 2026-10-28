from fastapi import APIRouter
from app.controllers.auth_controller import AuthController

router = APIRouter()

# Customer Authentication Routes
router.add_api_route("/login", AuthController.show_customer_login, methods=["GET"], name="customer_login")
router.add_api_route("/login", AuthController.handle_customer_login, methods=["POST"], name="customer_login_post")
router.add_api_route("/auth/login", AuthController.show_customer_login, methods=["GET"], name="customer_login_alias")
router.add_api_route("/auth/login", AuthController.handle_customer_login, methods=["POST"], name="customer_login_alias_post")
router.add_api_route("/register", AuthController.show_customer_register, methods=["GET"], name="customer_register")
router.add_api_route("/register", AuthController.handle_customer_register, methods=["POST"], name="customer_register_post")
router.add_api_route("/auth/register", AuthController.show_customer_register, methods=["GET"], name="customer_register_alias")
router.add_api_route("/auth/register", AuthController.handle_customer_register, methods=["POST"], name="customer_register_alias_post")

# OTP Verification Routes (Step 2 of registration)
router.add_api_route("/auth/verify-otp", AuthController.show_verify_otp, methods=["GET"], name="verify_otp")
router.add_api_route("/auth/verify-otp", AuthController.handle_verify_otp, methods=["POST"], name="verify_otp_post")
router.add_api_route("/auth/resend-otp", AuthController.resend_otp, methods=["POST"], name="resend_otp")

# Admin Authentication Routes
router.add_api_route("/admin/login", AuthController.show_admin_login, methods=["GET"], name="admin_login")
router.add_api_route("/admin/login", AuthController.handle_admin_login, methods=["POST"], name="admin_login_post")

# Logout
router.add_api_route("/logout", AuthController.handle_logout, methods=["GET"], name="logout")
router.add_api_route("/logout", AuthController.handle_logout, methods=["POST"], name="logout_post")
router.add_api_route("/auth/logout", AuthController.handle_logout, methods=["GET"], name="auth_logout_alias")
router.add_api_route("/auth/logout", AuthController.handle_logout, methods=["POST"], name="auth_logout_post_alias")
