import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import AnalysisResponse
from services.ai_analysis import analyze_with_claude
from services.pose_detection import run_pose_detection

router = APIRouter()


@router.post("/overhead-squat", response_model=AnalysisResponse)
async def analyze_overhead_squat(video: UploadFile = File(...), rotation: int = Form(0)):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        reps, annotated_path = run_pose_detection(tmp_path, exercise="overhead_squat", rotation=rotation)
        if not reps:
            raise HTTPException(status_code=422, detail="No reps detected in video")
        ai_feedback = analyze_with_claude(reps, exercise="overhead squat")
        return AnalysisResponse(
            exercise="overhead_squat",
            reps=reps,
            ai_feedback=ai_feedback,
            processed_video_url=f"/api/video/{os.path.basename(annotated_path)}",
        )
    finally:
        os.unlink(tmp_path)
