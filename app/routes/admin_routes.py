from fastapi import APIRouter
from app.controllers.admin_controller import AdminController

router = APIRouter(prefix="/admin")

router.add_api_route("/dashboard", AdminController.dashboard, methods=["GET"], name="admin_dashboard")

# User Management Routes
router.add_api_route("/users", AdminController.users_index, methods=["GET"], name="admin_users")
router.add_api_route("/users/create", AdminController.user_create_view, methods=["GET"], name="admin_user_create_view")
router.add_api_route("/users", AdminController.user_store, methods=["POST"], name="admin_user_store")
router.add_api_route("/users/{user_id}/edit", AdminController.user_edit_view, methods=["GET"], name="admin_user_edit_view")
router.add_api_route("/users/{user_id}/edit", AdminController.user_update, methods=["POST"], name="admin_user_update")
router.add_api_route("/users/{user_id}/delete", AdminController.user_delete, methods=["POST"], name="admin_user_delete")

# Monitoring Routes
router.add_api_route("/scans", AdminController.scans_index, methods=["GET"], name="admin_scans")
router.add_api_route("/logs", AdminController.logs_index, methods=["GET"], name="admin_logs")

# Product Management Routes
router.add_api_route("/products", AdminController.products_index, methods=["GET"], name="admin_products")
router.add_api_route("/products/create", AdminController.product_create_view, methods=["GET"], name="admin_product_create_view")
router.add_api_route("/products", AdminController.product_store, methods=["POST"], name="admin_product_store")
router.add_api_route("/products/{product_id}/edit", AdminController.product_edit_view, methods=["GET"], name="admin_product_edit_view")
router.add_api_route("/products/{product_id}/edit", AdminController.product_update, methods=["POST"], name="admin_product_update")
router.add_api_route("/products/{product_id}/delete", AdminController.product_delete, methods=["POST"], name="admin_product_delete")
# AI Copilot & Auto-Insert Routes
router.add_api_route("/ai-copilot", AdminController.ai_copilot_view, methods=["GET"], name="admin_ai_copilot")
router.add_api_route("/ai-copilot/sync-rag", AdminController.sync_rag_knowledge, methods=["POST"], name="admin_sync_rag")
router.add_api_route("/products/auto-insert", AdminController.product_auto_insert, methods=["POST"], name="admin_product_auto_insert")

# System Settings & AI Limits Routes
router.add_api_route("/settings", AdminController.settings_view, methods=["GET"], name="admin_settings")
router.add_api_route("/settings", AdminController.settings_update, methods=["POST"], name="admin_settings_update")

# External Connection Testing Hub Routes
router.add_api_route("/settings/test-email", AdminController.test_email_connection, methods=["POST"], name="admin_test_email")
router.add_api_route("/settings/test-wa", AdminController.test_wa_connection, methods=["POST"], name="admin_test_wa")
router.add_api_route("/settings/test-ai", AdminController.test_ai_connection, methods=["POST"], name="admin_test_ai")
router.add_api_route("/settings/test-db", AdminController.test_db_connection, methods=["POST"], name="admin_test_db")

# Admin Profile Routes
router.add_api_route("/profile", AdminController.profile_view, methods=["GET"], name="admin_profile")
router.add_api_route("/profile", AdminController.profile_update, methods=["POST"], name="admin_profile_update")
