import os
import tempfile
import uuid
import cv2
import av
import numpy as np
from ultralytics import YOLO
from models.schemas import RepData

os.environ.setdefault("YOLO_CONFIG_DIR", "/app/models")

MODEL_DIR = "/app/models"
MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n-pose.pt")
_model = None

ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def get_model():
    global _model
    if _model is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        _model = YOLO(MODEL_PATH)
    return _model


def _calc_angle(a, b, c) -> float:
    import math
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def _draw_overlay(frame: np.ndarray, angles: dict, rep_count: int, rep_history: list) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = w / 640
    th = max(1, int(scale * 1.5))
    fs = scale * 0.7

    def put(text, x, y, color):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th)

    y = int(h * 0.08)
    dy = int(h * 0.055)
    x = int(w * 0.03)

    lk, rk = angles.get("left_knee"), angles.get("right_knee")
    lh, rh = angles.get("left_hip"),  angles.get("right_hip")
    ls, rs = angles.get("left_shoulder"), angles.get("right_shoulder")

    if lk is not None: put(f"L-Knee: {lk:.1f}", x, y, (0, 255, 0));   y += dy
    if rk is not None: put(f"R-Knee: {rk:.1f}", x, y, (0, 255, 0));   y += dy
    if lh is not None: put(f"L-Hip:  {lh:.1f}", x, y, (0, 255, 255)); y += dy
    if rh is not None: put(f"R-Hip:  {rh:.1f}", x, y, (0, 255, 255)); y += dy
    if ls is not None: put(f"L-Shldr:{ls:.1f}", x, y, (255,180,0));   y += dy
    if rs is not None: put(f"R-Shldr:{rs:.1f}", x, y, (255,180,0));   y += dy
    put(f"Reps: {rep_count}", x, y, (80, 80, 255))

    rx = int(w * 0.6); ry = int(h * 0.08); rdy = int(h * 0.048)
    put("Rep History:", rx, ry, (255, 255, 255)); ry += rdy
    for rep in rep_history[-3:]:
        put(f"Rep {rep['rep_number']}:", rx, ry, (255, 255, 255)); ry += rdy
        if rep.get("left_hip_angle"):  put(f"  L-Hip: {rep['left_hip_angle']:.1f}",  rx, ry, (0,255,255)); ry += rdy
        if rep.get("right_hip_angle"): put(f"  R-Hip: {rep['right_hip_angle']:.1f}", rx, ry, (0,255,255)); ry += rdy
        if rep.get("left_knee_angle"): put(f"  L-Knee:{rep['left_knee_angle']:.1f}", rx, ry, (0,255,0));   ry += rdy
        if rep.get("right_knee_angle"):put(f"  R-Knee:{rep['right_knee_angle']:.1f}",rx, ry, (0,255,0));   ry += rdy
        if rep.get("duration_seconds"):put(f"  Dur:   {rep['duration_seconds']:.1f}s",rx,ry,(200,200,200));ry += rdy
        ry += int(rdy * 0.3)
    return frame


def _write_h264(frames: list, out_path: str, fps: int = 4) -> None:
    with av.open(out_path, "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.pix_fmt = "yuv420p"
        h, w = frames[0].shape[:2]
        stream.width  = w if w % 2 == 0 else w - 1
        stream.height = h if h % 2 == 0 else h - 1
        for bgr in frames:
            rgb = bgr[:stream.height, :stream.width, ::-1]
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def run_pose_detection(video_path: str, exercise: str, rotation: int = 0) -> tuple[list[RepData], str]:
    model = get_model()

    # Auto-detect rotation from metadata if not manually set.
    # We must disable OpenCV's auto-rotation (CAP_PROP_ORIENTATION_AUTO=0) before
    # reading metadata — otherwise Mac/FFmpeg auto-applies the rotation and
    # CAP_PROP_ORIENTATION_META still returns the original value, causing double rotation.
    if rotation == 0:
        cap_check = cv2.VideoCapture(video_path)
        cap_check.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        meta = int(cap_check.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
        cap_check.release()
        rotation = meta

    rotate_code = ROTATE_MAP.get(rotation)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable auto-rotation so we control it
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    keep_every = max(1, round(src_fps / 4))

    # Rep tracking state
    phase = "standing"
    rep_count = 0
    rep_history = []
    cur: dict = {}
    max_standing: dict = {}
    frame_idx = 0
    annotated_frames = []

    HINGE_THRESHOLD = 120;  LOCKOUT_THRESHOLD = 160
    SQUAT_THRESHOLD  = 120; STAND_THRESHOLD   = 160

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % keep_every != 0:
            frame_idx += 1
            continue

        # Apply rotation directly to the raw frame — no intermediate file
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        results = model.track(frame, persist=True, verbose=False)
        r = results[0]
        base = r.plot()

        angles: dict = {}
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            kp = r.keypoints.xy[0].tolist()
            if len(kp) >= 17:
                lh, rh = kp[11], kp[12]
                lk, rk = kp[13], kp[14]
                la, ra = kp[15], kp[16]
                ls, rs = kp[5],  kp[6]

                angles["left_hip"]   = _calc_angle(ls, lh, lk)
                angles["right_hip"]  = _calc_angle(rs, rh, rk)
                angles["left_knee"]  = _calc_angle(lh, lk, la)
                angles["right_knee"] = _calc_angle(rh, rk, ra)

                if exercise == "overhead_squat" and len(kp) > 10:
                    lwr, rwr = kp[9], kp[10]
                    angles["left_shoulder"]  = _calc_angle(lh, ls, lwr)
                    angles["right_shoulder"] = _calc_angle(rh, rs, rwr)

                avg_hip   = (angles["left_hip"] + angles["right_hip"]) / 2
                primary   = avg_hip if exercise == "deadlift" else (angles["left_knee"] + angles["right_knee"]) / 2
                down_thr  = HINGE_THRESHOLD if exercise == "deadlift" else SQUAT_THRESHOLD
                up_thr    = LOCKOUT_THRESHOLD if exercise == "deadlift" else STAND_THRESHOLD

                if phase == "standing":
                    for k, v in angles.items():
                        if k not in max_standing or v > max_standing[k]:
                            max_standing[k] = v
                    if primary < down_thr:
                        phase = "lowering"
                        cur = {**angles, "start_frame": frame_idx, "primary_min": primary}

                elif phase == "lowering":
                    if primary < cur.get("primary_min", primary):
                        cur["primary_min"] = primary
                        for k, v in angles.items():
                            cur[k] = v
                    if primary >= up_thr:
                        phase = "standing"
                        rep_count += 1
                        dur = (frame_idx - cur.get("start_frame", frame_idx)) / src_fps
                        rep_history.append({
                            "rep_number":        rep_count,
                            "left_hip_angle":    cur.get("left_hip"),
                            "right_hip_angle":   cur.get("right_hip"),
                            "left_knee_angle":   cur.get("left_knee"),
                            "right_knee_angle":  cur.get("right_knee"),
                            "left_shoulder_angle":  cur.get("left_shoulder"),
                            "right_shoulder_angle": cur.get("right_shoulder"),
                            "duration_seconds":  dur,
                            "left_hip_rom":  (max_standing.get("left_hip",  0) - (cur.get("left_hip")  or 0)) or None,
                            "right_hip_rom": (max_standing.get("right_hip", 0) - (cur.get("right_hip") or 0)) or None,
                            "left_knee_rom":  (max_standing.get("left_knee",  0) - (cur.get("left_knee")  or 0)) or None,
                            "right_knee_rom": (max_standing.get("right_knee", 0) - (cur.get("right_knee") or 0)) or None,
                        })
                        max_standing = {}
                        cur = {}

        annotated = _draw_overlay(base, angles, rep_count, rep_history)
        annotated_frames.append(annotated)
        frame_idx += 1

    cap.release()

    out_path = os.path.join(tempfile.gettempdir(), f"formiq_{uuid.uuid4().hex}.mp4")
    if annotated_frames:
        _write_h264(annotated_frames, out_path)

    reps = [
        RepData(
            rep_number=r["rep_number"],
            duration_seconds=r["duration_seconds"],
            left_hip_angle=r.get("left_hip_angle"),
            right_hip_angle=r.get("right_hip_angle"),
            left_knee_angle=r.get("left_knee_angle"),
            right_knee_angle=r.get("right_knee_angle"),
            left_shoulder_angle=r.get("left_shoulder_angle"),
            right_shoulder_angle=r.get("right_shoulder_angle"),
            left_hip_rom=r.get("left_hip_rom"),
            right_hip_rom=r.get("right_hip_rom"),
            left_knee_rom=r.get("left_knee_rom"),
            right_knee_rom=r.get("right_knee_rom"),
        )
        for r in rep_history
    ]
    return reps, out_path
