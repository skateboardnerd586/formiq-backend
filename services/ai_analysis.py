import os
import anthropic
import json
from models.schemas import RepData, AIFeedback


def build_rep_summary(reps: list[RepData], exercise: str) -> str:
    lines = [f"Exercise: {exercise}", f"Total reps: {len(reps)}", ""]
    for rep in reps:
        lines.append(f"Rep {rep.rep_number} ({rep.duration_seconds:.1f}s):")
        if rep.left_hip_angle is not None:
            lines.append(f"  Hip angles at peak: L={rep.left_hip_angle:.1f}° R={rep.right_hip_angle:.1f}°")
        if rep.left_knee_angle is not None:
            lines.append(f"  Knee angles at peak: L={rep.left_knee_angle:.1f}° R={rep.right_knee_angle:.1f}°")
        if rep.left_shoulder_angle is not None:
            lines.append(f"  Shoulder angles at peak: L={rep.left_shoulder_angle:.1f}° R={rep.right_shoulder_angle:.1f}°")
        if rep.left_hip_rom is not None:
            lines.append(f"  Hip ROM: L={rep.left_hip_rom:.1f}° R={rep.right_hip_rom:.1f}°")
        if rep.left_knee_rom is not None:
            lines.append(f"  Knee ROM: L={rep.left_knee_rom:.1f}° R={rep.right_knee_rom:.1f}°")
        if rep.left_shoulder_rom is not None:
            lines.append(f"  Shoulder ROM: L={rep.left_shoulder_rom:.1f}° R={rep.right_shoulder_rom:.1f}°")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert strength and conditioning coach and biomechanics specialist.
You receive joint angle and range-of-motion data extracted from video via pose estimation.
You respond ONLY with valid JSON — no prose, no markdown fences.
Be specific and data-driven. Reference actual angle values in your feedback.
A good hip hinge deadlift has ~45-60° hip angle at lockout and 70-100° hip ROM.
A good overhead squat has knee angles <100° at bottom, bilateral symmetry within 5°."""


def analyze_with_claude(reps: list[RepData], exercise: str) -> AIFeedback:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    client = anthropic.Anthropic(api_key=api_key)

    rep_summary = build_rep_summary(reps, exercise)

    prompt = f"""Analyze this {exercise} data and return a JSON object with exactly these fields:

{{
  "overall_score": <integer 1-10>,
  "summary": "<2-3 sentence overall assessment referencing specific angle values>",
  "key_issues": ["<specific issue with angle data cited>", ...],
  "corrective_exercises": ["<exercise name: why it helps>", ...],
  "rep_by_rep": ["<Rep 1: specific observation>", ...]
}}

Data:
{rep_summary}

Rules:
- overall_score: 1-10 based on form quality
- key_issues: max 3, each must cite a specific angle or ROM value from the data
- corrective_exercises: max 4, each tied to a specific issue found
- rep_by_rep: one entry per rep, note any deviation from the average
- If bilateral asymmetry > 10°, always flag it
- Return only the JSON object, nothing else"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    data = json.loads(message.content[0].text)
    return AIFeedback(**data)
