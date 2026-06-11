"""
analyzer.py — Multi-Exercise Technique Analyzer

Supports: Push-Up, Squat, Plank, Lunge

Each exercise uses MediaPipe Pose to detect body landmarks, calculates the
relevant joint angles, and returns beginner-friendly feedback (strengths,
tips, overall verdict) in English or Spanish.

Accuracy notes (why results are trustworthy):
  - BOTH the left and right side are tracked; the more clearly visible side
    is chosen automatically, so it doesn't matter which way the user faces.
  - Low-confidence frames (a joint hidden or off-screen) are discarded using
    MediaPipe's per-landmark visibility score.
  - "Deepest" positions use the 10th/90th percentile instead of the raw
    min()/max(), so a single jittery frame can't fake good depth.
  - A range-of-motion check confirms a real repetition actually happened,
    so a static or unrelated video is not graded as "great".
"""

import base64
import os
import math
import shutil
import subprocess
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────
# Model setup — download automatically if missing
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


def ensure_model():
    """Download the pose landmarker model if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (~14 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# Landmark indices for each side of the body
SIDES = {
    "left":  {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28},
}

# A landmark below this visibility is considered unreliable
MIN_VISIBILITY = 0.5
# Need at least this many usable frames to judge form
MIN_VALID_FRAMES = 6

# Joints each exercise actually depends on
NEEDED_JOINTS = {
    "pushup": ["shoulder", "elbow", "wrist", "hip", "ankle"],
    "squat":  ["shoulder", "hip", "knee", "ankle"],
    "plank":  ["shoulder", "hip", "ankle"],
    "lunge":  ["shoulder", "hip", "knee", "ankle"],
}

# Joint pairs to draw as skeleton lines in the annotated video
SKELETON_CONNECTIONS = {
    "pushup": [("shoulder","elbow"),("elbow","wrist"),("shoulder","hip"),("hip","ankle")],
    "squat":  [("shoulder","hip"),("hip","knee"),("knee","ankle")],
    "plank":  [("shoulder","hip"),("hip","ankle")],
    "lunge":  [("shoulder","hip"),("hip","knee"),("knee","ankle")],
}

# Per-exercise angle display configuration for the video overlay and chart
ANGLE_CONFIG = {
    "pushup": {
        "primary_joint": "elbow",   "primary_label": "Elbow",
        "primary_threshold": 95.0,  "primary_dir": "max",
        "secondary_joint": "hip",   "secondary_label": "Body Line",
        "secondary_threshold": 160.0, "secondary_dir": "min",
    },
    "squat": {
        "primary_joint": "knee",    "primary_label": "Knee",
        "primary_threshold": 100.0, "primary_dir": "max",
        "secondary_joint": "shoulder", "secondary_label": "Torso Lean",
        "secondary_threshold": 50.0,   "secondary_dir": "max",
    },
    "plank": {
        "primary_joint": "hip",     "primary_label": "Body Line",
        "primary_threshold": 160.0, "primary_dir": "min",
    },
    "lunge": {
        "primary_joint": "knee",    "primary_label": "Knee",
        "primary_threshold": 110.0, "primary_dir": "max",
        "secondary_joint": "shoulder", "secondary_label": "Torso Lean",
        "secondary_threshold": 30.0,   "secondary_dir": "max",
    },
}

# Drawing colors (BGR)
_SKEL_COLOR  = (100, 220, 255)  # warm yellow-ish
_JOINT_COLOR = (255, 255, 255)
_GOOD_COLOR  = (50,  200,  50)  # green
_BAD_COLOR   = (50,   50, 220)  # red


# ══════════════════════════════════════════════
# Localized feedback messages
# ══════════════════════════════════════════════
MSG = {
    "en": {
        "no_pose": "We couldn't detect your body in the video. Make sure your whole body is visible and the room is well lit.",
        "low_visibility": "We could see you, but not clearly enough to judge your form. Try filming from the side with your whole body in frame and good lighting.",
        "pushup": {
            "depth_good": "You lowered yourself with good depth — your chest came down nice and close to the floor.",
            "depth_bad": "Try lowering yourself further. Bend your arms until your chest is close to the ground for a fuller push-up.",
            "align_good": "Your body held a strong, straight line from your head to your heels.",
            "align_bad": "Keep your body straight from head to heels. Squeeze your core and glutes so your hips don't sag down or pike up.",
            "norep": "We couldn't clearly see a full push-up. Film yourself from the side, keep your whole body in frame, and complete the full up-and-down movement.",
            "great": "Excellent push-up! Your depth and body line both look solid.",
            "okay": "Good effort! A small adjustment or two will make your push-ups even better.",
            "needs": "Keep practicing! Focus on the tips below and your push-ups will improve quickly.",
        },
        "squat": {
            "depth_good": "Great depth — you squatted low enough to get real work from your legs.",
            "depth_bad": "Try sitting a little deeper, as if lowering into a chair, until your thighs reach about parallel with the floor.",
            "back_good": "Your torso stayed tall and controlled throughout the movement.",
            "back_bad": "Keep your chest up and avoid leaning too far forward. Push your hips back and look straight ahead as you go down.",
            "norep": "We couldn't clearly see a full squat. Film yourself from the side, keep your whole body in frame, and complete the full movement.",
            "great": "Solid squat! You're hitting good depth with a strong, upright posture.",
            "okay": "Nice work! A small tweak will take your squat to the next level.",
            "needs": "Keep at it! Focus on the tips below to build a stronger squat.",
        },
        "plank": {
            "align_good": "Your body held a strong straight line — great plank position!",
            "align_bad": "Aim for a straight line from your head to your heels. Don't let your hips drop down or lift up too high.",
            "stable_good": "You held the position steady with very little wobble.",
            "stable_bad": "Try to stay as still as possible. Brace your core and breathe slowly to stay steady.",
            "great": "Strong plank! Your alignment and stability are both on point.",
            "okay": "Good effort! A small adjustment will perfect your plank.",
            "needs": "Planks are tough! Focus on the tips below to build a stronger hold.",
        },
        "lunge": {
            "depth_good": "Nice depth — your front leg bent enough for an effective lunge.",
            "depth_bad": "Step out a little further and bend your front knee toward a right angle, keeping it above your ankle.",
            "torso_good": "Your upper body stayed tall and upright through the lunge.",
            "torso_bad": "Keep your torso upright — imagine a string pulling you up from the top of your head — instead of leaning forward.",
            "norep": "We couldn't clearly see a full lunge. Film yourself from the side, keep your whole body in frame, and complete the full movement.",
            "great": "Nice lunge! Good depth with a tall, balanced posture.",
            "okay": "Getting there! A few tweaks and your lunges will be textbook.",
            "needs": "Lunges take practice! Follow the tips below to improve your form.",
        },
    },
    "es": {
        "no_pose": "No pudimos detectar tu cuerpo en el video. Asegúrate de que se vea tu cuerpo completo y de que haya buena luz.",
        "low_visibility": "Te vimos, pero no con suficiente claridad para evaluar tu técnica. Intenta grabar de lado, con todo el cuerpo en cuadro y buena iluminación.",
        "pushup": {
            "depth_good": "Bajaste con buena profundidad: tu pecho llegó bien cerca del suelo.",
            "depth_bad": "Intenta bajar más. Dobla los brazos hasta que tu pecho quede cerca del suelo para una flexión más completa.",
            "align_good": "Tu cuerpo mantuvo una línea recta y firme desde la cabeza hasta los talones.",
            "align_bad": "Mantén el cuerpo recto de la cabeza a los talones. Aprieta el abdomen y los glúteos para que la cadera no se hunda ni se levante.",
            "norep": "No pudimos ver una flexión completa con claridad. Grábate de lado, mantén todo el cuerpo en cuadro y completa todo el movimiento de bajada y subida.",
            "great": "¡Excelente flexión! Tu profundidad y la línea del cuerpo se ven muy bien.",
            "okay": "¡Buen trabajo! Uno o dos pequeños ajustes harán tus flexiones aún mejores.",
            "needs": "¡Sigue practicando! Concéntrate en los consejos de abajo y tus flexiones mejorarán rápido.",
        },
        "squat": {
            "depth_good": "Buena profundidad: bajaste lo suficiente para trabajar bien las piernas.",
            "depth_bad": "Intenta bajar un poco más, como si te sentaras en una silla, hasta que tus muslos queden casi paralelos al suelo.",
            "back_good": "Tu torso se mantuvo erguido y controlado durante todo el movimiento.",
            "back_bad": "Mantén el pecho arriba y evita inclinarte demasiado hacia adelante. Lleva la cadera hacia atrás y mira al frente al bajar.",
            "norep": "No pudimos ver una sentadilla completa con claridad. Grábate de lado, mantén todo el cuerpo en cuadro y completa todo el movimiento.",
            "great": "¡Sentadilla sólida! Llegas a buena profundidad con una postura erguida y fuerte.",
            "okay": "¡Buen trabajo! Un pequeño ajuste llevará tu sentadilla al siguiente nivel.",
            "needs": "¡Sigue así! Concéntrate en los consejos de abajo para una sentadilla más fuerte.",
        },
        "plank": {
            "align_good": "Tu cuerpo mantuvo una línea recta y firme: ¡excelente posición de plancha!",
            "align_bad": "Busca una línea recta de la cabeza a los talones. No dejes que la cadera se hunda ni se levante demasiado.",
            "stable_good": "Mantuviste la posición estable, con muy poco movimiento.",
            "stable_bad": "Intenta quedarte lo más quieto posible. Aprieta el abdomen y respira despacio para mantener la estabilidad.",
            "great": "¡Plancha fuerte! Tu alineación y estabilidad están perfectas.",
            "okay": "¡Buen esfuerzo! Un pequeño ajuste perfeccionará tu plancha.",
            "needs": "¡Las planchas son difíciles! Concéntrate en los consejos de abajo para aguantar mejor.",
        },
        "lunge": {
            "depth_good": "Buena profundidad: tu pierna delantera se dobló lo suficiente para una zancada efectiva.",
            "depth_bad": "Da un paso un poco más largo y dobla la rodilla delantera hacia un ángulo recto, manteniéndola sobre el tobillo.",
            "torso_good": "Tu parte superior se mantuvo alta y erguida durante la zancada.",
            "torso_bad": "Mantén el torso erguido, como si un hilo te tirara desde la cabeza, en lugar de inclinarte hacia adelante.",
            "norep": "No pudimos ver una zancada completa con claridad. Grábate de lado, mantén todo el cuerpo en cuadro y completa todo el movimiento.",
            "great": "¡Buena zancada! Buena profundidad con una postura alta y equilibrada.",
            "okay": "¡Vas por buen camino! Unos ajustes y tus zancadas serán de manual.",
            "needs": "¡Las zancadas requieren práctica! Sigue los consejos de abajo para mejorar tu técnica.",
        },
    },
}


def _m(language: str, exercise: str, key: str) -> str:
    """Fetch a localized message, falling back to English."""
    lang = language if language in MSG else "en"
    try:
        return MSG[lang][exercise][key]
    except KeyError:
        return MSG["en"][exercise][key]


# ──────────────────────────────────────────────
# Extract body landmarks from every frame
# ──────────────────────────────────────────────
def extract_landmarks(video_path: str):
    """
    Run MediaPipe Pose on each frame and return (frames, fps).

    Each frame record:
        {"left":  {"shoulder": (x, y, vis), ...},
         "right": {"shoulder": (x, y, vis), ...},
         "frame_idx": int}   ← original frame number in the video
    """
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_frames = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                h, w, _ = frame.shape
                landmarks = result.pose_landmarks[0]
                record = {"frame_idx": frame_idx}
                for side, joints in SIDES.items():
                    record[side] = {}
                    for name, idx in joints.items():
                        lm = landmarks[idx]
                        vis = getattr(lm, "visibility", 1.0)
                        record[side][name] = (lm.x * w, lm.y * h, vis)
                all_frames.append(record)

            frame_idx += 1

    cap.release()
    return all_frames, float(fps)


# ──────────────────────────────────────────────
# Side selection + visibility filtering
# ──────────────────────────────────────────────
def _side_visibility(frames, side, needed) -> float:
    """Average visibility of the needed joints on one side, across all frames."""
    if not frames:
        return 0.0
    scores = []
    for rec in frames:
        joints = rec[side]
        scores.append(np.mean([joints[j][2] for j in needed]))
    return float(np.mean(scores))


def _clean_frames(frames, side, needed) -> list[dict]:
    """
    Keep only frames where every needed joint on the chosen side is visible
    enough, and return them as plain {name: (x, y)} dictionaries.
    The special key "__frame_idx" carries the original video frame number.
    """
    clean = []
    for rec in frames:
        joints = rec[side]
        if all(joints[j][2] >= MIN_VISIBILITY for j in needed):
            d = {j: (joints[j][0], joints[j][1]) for j in needed}
            d["__frame_idx"] = rec.get("frame_idx", len(clean))
            clean.append(d)
    return clean


def _prepare(frames, exercise):
    """
    Choose the clearer side and return (clean_frames, side_name).
    clean_frames may be empty if visibility was too low.
    """
    needed = NEEDED_JOINTS[exercise]
    left_vis = _side_visibility(frames, "left", needed)
    right_vis = _side_visibility(frames, "right", needed)
    side = "left" if left_vis >= right_vis else "right"
    return _clean_frames(frames, side, needed), side


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def calculate_angle(point_a, point_b, point_c) -> float:
    """Angle at point_b (degrees, 0-180) using the dot-product formula."""
    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    ba = a - b
    bc = c - b

    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = np.clip(np.dot(ba, bc) / (mag_ba * mag_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vertical_lean(top_point, bottom_point) -> float:
    """
    How far the segment bottom→top tilts away from straight-up (degrees).
    0° = perfectly vertical/upright. Used for torso posture checks.
    """
    dx = top_point[0] - bottom_point[0]
    dy = top_point[1] - bottom_point[1]  # image y grows downward
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0
    # "Up" in image space is (0, -1)
    cos_angle = np.clip((dy * -1.0) / mag, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _series(clean, fn):
    """Apply an angle function to every clean frame, returning a numpy array."""
    return np.array([fn(lm) for lm in clean], dtype=float)


# ══════════════════════════════════════════════
# Visual overlay helpers
# ══════════════════════════════════════════════
def _draw_angle_label(img, pos, text, is_good, dx=12, dy=-14):
    """Draw angle text with a solid dark background for readability."""
    color = _GOOD_COLOR if is_good else _BAD_COLOR
    tx, ty = pos[0] + dx, pos[1] + dy
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = 3
    cv2.rectangle(img,
                  (tx - pad, ty - th - pad),
                  (tx + tw + pad, ty + base + pad),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)


def _draw_frame_overlay(frame, joints, exercise, angle_data):
    """Return a copy of frame with skeleton lines, joint dots, and angle labels."""
    out = frame.copy()
    for j1, j2 in SKELETON_CONNECTIONS.get(exercise, []):
        if j1 in joints and j2 in joints:
            cv2.line(out, joints[j1], joints[j2], _SKEL_COLOR, 2, cv2.LINE_AA)
    for pos in joints.values():
        cv2.circle(out, pos, 6, _JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, pos, 6, (30, 30, 30), 1, cv2.LINE_AA)

    cfg = ANGLE_CONFIG.get(exercise, {})

    p_val = angle_data.get("primary")
    if p_val is not None:
        pj   = cfg.get("primary_joint")
        pt   = cfg.get("primary_threshold", 90)
        pdir = cfg.get("primary_dir", "max")
        good = (p_val <= pt) if pdir == "max" else (p_val >= pt)
        if pj in joints:
            _draw_angle_label(out, joints[pj], f"{p_val:.0f}\xb0", good, dx=10, dy=-14)

    s_val = angle_data.get("secondary")
    if s_val is not None:
        sj   = cfg.get("secondary_joint")
        st   = cfg.get("secondary_threshold", 50)
        sdir = cfg.get("secondary_dir", "max")
        good = (s_val <= st) if sdir == "max" else (s_val >= st)
        if sj in joints:
            _draw_angle_label(out, joints[sj], f"{s_val:.0f}\xb0", good, dx=10, dy=22)

    return out


def generate_annotated_video(video_path: str, exercise: str,
                             clean: list, angle_series: list) -> str:
    """
    Re-reads the source video, draws skeleton + angle overlays on frames that
    have landmark data, and returns the result as a base64-encoded MP4 string.
    """
    joints_by_frame = {}
    for lm in clean:
        fi = lm.get("__frame_idx")
        if fi is not None:
            joints_by_frame[fi] = {k: v for k, v in lm.items() if k != "__frame_idx"}

    angles_by_frame = {item["frame_idx"]: item for item in angle_series}

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scale = min(1.0, 640.0 / orig_w) if orig_w > 0 else 1.0
    out_w = max(2, int(orig_w * scale) & ~1)
    out_h = max(2, int(orig_h * scale) & ~1)
    out_fps = min(orig_fps, 20.0)
    frame_step = max(1, round(orig_fps / out_fps))

    out_path = video_path + "_annotated.mp4"

    # Try codecs in order; avc1 (H.264) plays best in browsers, mp4v as fallback
    writer = None
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        cap.release()
        raise RuntimeError("No usable video codec found (tried avc1, mp4v)")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                if scale < 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                if frame_idx in joints_by_frame:
                    scaled_joints = {
                        j: (int(x * scale), int(y * scale))
                        for j, (x, y) in joints_by_frame[frame_idx].items()
                    }
                    frame = _draw_frame_overlay(
                        frame, scaled_joints, exercise,
                        angles_by_frame.get(frame_idx, {})
                    )
                writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # Re-encode to H.264 so all browsers can play it
    if shutil.which("ffmpeg"):
        h264_path = out_path + ".h264.mp4"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", out_path,
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                 "-movflags", "+faststart", "-an", h264_path],
                capture_output=True, timeout=120,
            )
            if proc.returncode == 0 and os.path.getsize(h264_path) > 0:
                os.remove(out_path)
                out_path = h264_path
            else:
                print(f"[visual] ffmpeg failed: {proc.stderr.decode()[:200]}")
        except Exception as exc:
            print(f"[visual] ffmpeg exception: {exc}")

    try:
        with open(out_path, "rb") as fh:
            raw = fh.read()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass

    # Skip sending video if it's too large (>8 MB base64 ≈ 6 MB raw)
    if len(raw) > 6 * 1024 * 1024:
        print(f"[visual] video too large ({len(raw)//1024} KB), skipping video embed")
        return ""

    return base64.b64encode(raw).decode("utf-8")


def _build_angle_timeline(angle_series: list, exercise: str) -> dict:
    """Return chart-ready angle timeline data for the frontend."""
    cfg = ANGLE_CONFIG.get(exercise, {})
    frames    = [item["frame_idx"] for item in angle_series]
    primary   = [item.get("primary")   for item in angle_series]
    secondary = [item.get("secondary") for item in angle_series]

    timeline = {
        "frames":            frames,
        "primary":           primary,
        "primary_label":     cfg.get("primary_label", "Angle"),
        "primary_threshold": cfg.get("primary_threshold"),
        "primary_dir":       cfg.get("primary_dir", "max"),
    }
    if any(s is not None for s in secondary):
        timeline.update({
            "secondary":           secondary,
            "secondary_label":     cfg.get("secondary_label", "Angle 2"),
            "secondary_threshold": cfg.get("secondary_threshold"),
            "secondary_dir":       cfg.get("secondary_dir", "max"),
        })
    return timeline


# ──────────────────────────────────────────────
# Main analysis dispatcher
# ──────────────────────────────────────────────
def analyze(video_path: str, exercise: str, language: str = "en") -> dict:
    """
    Analyse a video for the given exercise type.
    Returns beginner-friendly results with strengths and tips.
    """
    if language not in MSG:
        language = "en"

    frames, _fps = extract_landmarks(video_path)

    if not frames:
        return _unknown(exercise, language, MSG[language]["no_pose"])

    clean, side = _prepare(frames, exercise)

    if len(clean) < MIN_VALID_FRAMES:
        return _unknown(exercise, language, MSG[language]["low_visibility"], len(clean))

    analyzers = {
        "pushup": _analyze_pushup,
        "squat":  _analyze_squat,
        "plank":  _analyze_plank,
        "lunge":  _analyze_lunge,
    }
    analyzer_fn = analyzers.get(exercise, _analyze_pushup)
    result = analyzer_fn(clean, exercise, language)
    result["side_used"] = side

    angle_series = result.pop("_angle_series", None)
    if angle_series:
        try:
            result["annotated_video"] = generate_annotated_video(
                video_path, exercise, clean, angle_series
            )
            result["angle_timeline"] = _build_angle_timeline(angle_series, exercise)
        except Exception as exc:
            print(f"[visual] annotation skipped: {exc}")

    return result


def _unknown(exercise, language, message, frames=0) -> dict:
    return {
        "exercise": exercise,
        "overall": "unknown",
        "strengths": [],
        "tips": [message],
        "summary": message,
        "frames_analyzed": frames,
    }


def _verdict(good_count: int, total: int) -> str:
    """great = all criteria met, needs work = none met, otherwise okay."""
    if good_count >= total:
        return "great"
    if good_count == 0:
        return "needs work"
    return "okay"


# ──────────────────────────────────────────────
# Push-Up Analysis
# ──────────────────────────────────────────────
# Thresholds (degrees) — chosen from beginner push-up biomechanics
PU_DEPTH_MAX = 95.0      # bottom elbow bend must reach at least this
PU_ALIGN_MIN = 160.0     # shoulder–hip–ankle line must stay this straight
PU_MIN_ROM = 30.0        # elbow must travel this much to count as a real rep


def _analyze_pushup(clean, exercise, language) -> dict:
    elbow = _series(clean, lambda lm: calculate_angle(lm["shoulder"], lm["elbow"], lm["wrist"]))
    body = _series(clean, lambda lm: calculate_angle(lm["shoulder"], lm["hip"], lm["ankle"]))

    bottom_elbow = float(np.percentile(elbow, 10))   # deepest (robust)
    top_elbow = float(np.percentile(elbow, 90))      # most extended
    elbow_rom = top_elbow - bottom_elbow
    body_align = float(np.median(body))

    did_rep = elbow_rom >= PU_MIN_ROM

    # If no real push-up motion is detected, don't pretend to grade it.
    if not did_rep:
        res = _unknown(exercise, language, _m(language, "pushup", "norep"), len(clean))
        res["min_elbow_angle"] = round(bottom_elbow, 1)
        res["avg_body_alignment"] = round(body_align, 1)
        fi = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
        res["_angle_series"] = [
            {"frame_idx": int(fi[i]), "primary": float(elbow[i]), "secondary": float(body[i])}
            for i in range(len(clean))
        ]
        return res

    depth_good = bottom_elbow <= PU_DEPTH_MAX
    align_good = body_align >= PU_ALIGN_MIN

    strengths, tips = [], []
    (strengths if depth_good else tips).append(_m(language, "pushup", "depth_good" if depth_good else "depth_bad"))
    (strengths if align_good else tips).append(_m(language, "pushup", "align_good" if align_good else "align_bad"))

    overall = _verdict(sum([depth_good, align_good]), 2)
    summary = _m(language, "pushup", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    frame_indices = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "min_elbow_angle": round(bottom_elbow, 1),
        "avg_body_alignment": round(body_align, 1),
        "_angle_series": [
            {"frame_idx": int(frame_indices[i]), "primary": float(elbow[i]), "secondary": float(body[i])}
            for i in range(len(clean))
        ],
    }


# ──────────────────────────────────────────────
# Squat Analysis
# ──────────────────────────────────────────────
SQ_DEPTH_MAX = 100.0     # knee bend at bottom (≈ parallel) or lower
SQ_LEAN_MAX = 50.0       # torso lean from vertical (forward lean limit)
SQ_MIN_ROM = 40.0        # knee must travel this much to count as a rep


def _analyze_squat(clean, exercise, language) -> dict:
    knee = _series(clean, lambda lm: calculate_angle(lm["hip"], lm["knee"], lm["ankle"]))
    lean = _series(clean, lambda lm: vertical_lean(lm["shoulder"], lm["hip"]))

    bottom_knee = float(np.percentile(knee, 10))
    top_knee = float(np.percentile(knee, 90))
    knee_rom = top_knee - bottom_knee
    # Torso lean measured at the deepest part of the squat (worst case)
    deep_idx = knee <= np.percentile(knee, 25)
    avg_lean = float(np.mean(lean[deep_idx])) if deep_idx.any() else float(np.mean(lean))

    did_rep = knee_rom >= SQ_MIN_ROM
    if not did_rep:
        res = _unknown(exercise, language, _m(language, "squat", "norep"), len(clean))
        fi = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
        res["_angle_series"] = [
            {"frame_idx": int(fi[i]), "primary": float(knee[i]), "secondary": float(lean[i])}
            for i in range(len(clean))
        ]
        return res

    depth_good = bottom_knee <= SQ_DEPTH_MAX
    back_good = avg_lean <= SQ_LEAN_MAX

    strengths, tips = [], []
    (strengths if depth_good else tips).append(_m(language, "squat", "depth_good" if depth_good else "depth_bad"))
    (strengths if back_good else tips).append(_m(language, "squat", "back_good" if back_good else "back_bad"))

    overall = _verdict(sum([depth_good, back_good]), 2)
    summary = _m(language, "squat", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    frame_indices = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "min_knee_angle": round(bottom_knee, 1),
        "avg_torso_lean": round(avg_lean, 1),
        "_angle_series": [
            {"frame_idx": int(frame_indices[i]), "primary": float(knee[i]), "secondary": float(lean[i])}
            for i in range(len(clean))
        ],
    }


# ──────────────────────────────────────────────
# Plank Analysis (a static hold — no repetition)
# ──────────────────────────────────────────────
PL_ALIGN_MIN = 160.0     # shoulder–hip–ankle straightness
PL_STABILITY_MAX = 0.05  # hip wobble as a fraction of body length


def _analyze_plank(clean, exercise, language) -> dict:
    body = _series(clean, lambda lm: calculate_angle(lm["shoulder"], lm["hip"], lm["ankle"]))

    # Body length (shoulder→ankle) normalises wobble across video resolutions
    body_lengths = _series(
        clean,
        lambda lm: math.hypot(lm["shoulder"][0] - lm["ankle"][0],
                              lm["shoulder"][1] - lm["ankle"][1]),
    )
    body_len = float(np.median(body_lengths)) or 1.0
    hip_y = _series(clean, lambda lm: lm["hip"][1])
    wobble = float(np.std(hip_y)) / body_len

    body_align = float(np.median(body))

    align_good = body_align >= PL_ALIGN_MIN
    stable_good = wobble <= PL_STABILITY_MAX

    strengths, tips = [], []
    (strengths if align_good else tips).append(_m(language, "plank", "align_good" if align_good else "align_bad"))
    (strengths if stable_good else tips).append(_m(language, "plank", "stable_good" if stable_good else "stable_bad"))

    overall = _verdict(sum([align_good, stable_good]), 2)
    summary = _m(language, "plank", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    frame_indices = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "avg_body_alignment": round(body_align, 1),
        "_angle_series": [
            {"frame_idx": int(frame_indices[i]), "primary": float(body[i])}
            for i in range(len(clean))
        ],
    }


# ──────────────────────────────────────────────
# Lunge Analysis
# ──────────────────────────────────────────────
LU_DEPTH_MAX = 110.0     # front knee bend at the bottom
LU_LEAN_MAX = 30.0       # torso should stay more upright than a squat
LU_MIN_ROM = 35.0        # knee travel to count as a real rep


def _analyze_lunge(clean, exercise, language) -> dict:
    knee = _series(clean, lambda lm: calculate_angle(lm["hip"], lm["knee"], lm["ankle"]))
    lean = _series(clean, lambda lm: vertical_lean(lm["shoulder"], lm["hip"]))

    bottom_knee = float(np.percentile(knee, 10))
    top_knee = float(np.percentile(knee, 90))
    knee_rom = top_knee - bottom_knee
    deep_idx = knee <= np.percentile(knee, 25)
    avg_lean = float(np.mean(lean[deep_idx])) if deep_idx.any() else float(np.mean(lean))

    did_rep = knee_rom >= LU_MIN_ROM
    if not did_rep:
        res = _unknown(exercise, language, _m(language, "lunge", "norep"), len(clean))
        fi = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
        res["_angle_series"] = [
            {"frame_idx": int(fi[i]), "primary": float(knee[i]), "secondary": float(lean[i])}
            for i in range(len(clean))
        ]
        return res

    depth_good = bottom_knee <= LU_DEPTH_MAX
    torso_good = avg_lean <= LU_LEAN_MAX

    strengths, tips = [], []
    (strengths if depth_good else tips).append(_m(language, "lunge", "depth_good" if depth_good else "depth_bad"))
    (strengths if torso_good else tips).append(_m(language, "lunge", "torso_good" if torso_good else "torso_bad"))

    overall = _verdict(sum([depth_good, torso_good]), 2)
    summary = _m(language, "lunge", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    frame_indices = [lm.get("__frame_idx", i) for i, lm in enumerate(clean)]
    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "min_knee_angle": round(bottom_knee, 1),
        "avg_torso_lean": round(avg_lean, 1),
        "_angle_series": [
            {"frame_idx": int(frame_indices[i]), "primary": float(knee[i]), "secondary": float(lean[i])}
            for i in range(len(clean))
        ],
    }
