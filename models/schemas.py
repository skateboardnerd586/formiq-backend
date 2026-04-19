from pydantic import BaseModel
from typing import Optional


class RepData(BaseModel):
    rep_number: int
    duration_seconds: float
    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    left_shoulder_angle: Optional[float] = None
    right_shoulder_angle: Optional[float] = None
    left_hip_rom: Optional[float] = None
    right_hip_rom: Optional[float] = None
    left_knee_rom: Optional[float] = None
    right_knee_rom: Optional[float] = None
    left_shoulder_rom: Optional[float] = None
    right_shoulder_rom: Optional[float] = None


class AIFeedback(BaseModel):
    overall_score: int  # 1-10
    summary: str
    key_issues: list[str]
    corrective_exercises: list[str]
    rep_by_rep: list[str]


class AnalysisResponse(BaseModel):
    exercise: str
    reps: list[RepData]
    ai_feedback: AIFeedback
    processed_video_url: Optional[str] = None
