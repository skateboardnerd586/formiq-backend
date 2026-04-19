import cv2
import tempfile
import os


TARGET_FPS = 4


def _read_rotation(video_path: str) -> int:
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            rotation_prop = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            cap.release()
            if rotation_prop > 0:
                return int(rotation_prop)
        return 0
    except Exception:
        return 0


def preprocess_video(video_path: str, manual_rotation: int = 0) -> str:
    """Downsample to TARGET_FPS, apply manual rotation, and strip metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path

    auto_rotation = _read_rotation(video_path)
    rotation = manual_rotation if manual_rotation != 0 else auto_rotation

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    keep_every = max(1, round(src_fps / TARGET_FPS))

    out_w, out_h = (height, width) if rotation in (90, 270) else (width, height)

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (out_w, out_h))

    rotate_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotate_map.get(rotation)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % keep_every == 0:
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path
