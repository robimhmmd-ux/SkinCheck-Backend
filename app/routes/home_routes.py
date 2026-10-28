from fastapi import APIRouter
from app.controllers.home_controller import HomeController

router = APIRouter()

router.add_api_route("/", HomeController.index, methods=["GET"], name="home")
router.add_api_route("/activity-diagram", HomeController.activity_diagram, methods=["GET"], name="activity_diagram")
router.add_api_route("/usecase-diagram", HomeController.usecase_diagram, methods=["GET"], name="usecase_diagram")
