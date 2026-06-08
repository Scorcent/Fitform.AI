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

import os
import math
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
        "no_pose": "No pudimos detectar tu cuerpo en el video. Asegurate de que se vea tu cuerpo completo y de que haya buena luz.",
        "low_visibility": "Te vimos, pero no con suficiente claridad para evaluar tu tecnica. Intenta grabar de lado, con todo el cuerpo en cuadro y buena iluminacion.",
        "pushup": {
            "depth_good": "Bajaste con buena profundidad: tu pecho llego bien cerca del suelo.",
            "depth_bad": "Intenta bajar mas. Dobla los brazos hasta que tu pecho quede cerca del suelo para una flexion mas completa.",
            "align_good": "Tu cuerpo mantuvo una linea recta y firme desde la cabeza hasta los talones.",
            "align_bad": "Manten el cuerpo recto de la cabeza a los talones. Aprieta el abdomen y los gluteos para que la cadera no se hunda ni se levante.",
            "norep": "No pudimos ver una flexion completa con claridad. Grabate de lado, manten todo el cuerpo en cuadro y completa todo el movimiento de bajada y subida.",
            "great": "Excelente flexion! Tu profundidad y la linea del cuerpo se ven muy bien.",
            "okay": "Buen trabajo! Uno o dos pequenos ajustes haran tus flexiones aun mejores.",
            "needs": "Sigue practicando! Concentrate en los consejos de abajo y tus flexiones mejoraran rapido.",
        },
        "squat": {
            "depth_good": "Buena profundidad: bajaste lo suficiente para trabajar bien las piernas.",
            "depth_bad": "Intenta bajar un poco mas, como si te sentaras en una silla, hasta que tus muslos queden casi paralelos al suelo.",
            "back_good": "Tu torso se mantuvo erguido y controlado durante todo el movimiento.",
            "back_bad": "Manten el pecho arriba y evita inclinarte demasiado hacia adelante. Lleva la cadera hacia atras y mira al frente al bajar.",
            "norep": "No pudimos ver una sentadilla completa con claridad. Grabate de lado, manten todo el cuerpo en cuadro y completa todo el movimiento.",
            "great": "Sentadilla solida! Llegas a buena profundidad con una postura erguida y fuerte.",
            "okay": "Buen trabajo! Un pequeno ajuste llevara tu sentadilla al siguiente nivel.",
            "needs": "Sigue asi! Concentrate en los consejos de abajo para una sentadilla mas fuerte.",
        },
        "plank": {
            "align_good": "Tu cuerpo mantuvo una linea recta y firme: excelente posicion de plancha!",
            "align_bad": "Busca una linea recta de la cabeza a los talones. No dejes que la cadera se hunda ni se levante demasiado.",
            "stable_good": "Mantuviste la posicion estable, con muy poco movimiento.",
            "stable_bad": "Intenta quedarte lo mas quieto posible. Aprieta el abdomen y respira despacio para mantener la estabilidad.",
            "great": "Plancha fuerte! Tu alineacion y estabilidad estan perfectas.",
            "okay": "Buen esfuerzo! Un pequeno ajuste perfeccionara tu plancha.",
            "needs": "Las planchas son dificiles! Concentrate en los consejos de abajo para aguantar mejor.",
        },
        "lunge": {
            "depth_good": "Buena profundidad: tu pierna delantera se doblo lo suficiente para una zancada efectiva.",
            "depth_bad": "Da un paso un poco mas largo y dobla la rodilla delantera hacia un angulo recto, manteniendola sobre el tobillo.",
            "torso_good": "Tu parte superior se mantuvo alta y erguida durante la zancada.",
            "torso_bad": "Manten el torso erguido, como si un hilo te tirara desde la cabeza, en lugar de inclinarte hacia adelante.",
            "norep": "No pudimos ver una zancada completa con claridad. Grabate de lado, manten todo el cuerpo en cuadro y completa todo el movimiento.",
            "great": "Buena zancada! Buena profundidad con una postura alta y equilibrada.",
            "okay": "Vas por buen camino! Unos ajustes y tus zancadas seran de manual.",
            "needs": "Las zancadas requieren practica! Sigue los consejos de abajo para mejorar tu tecnica.",
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
def extract_landmarks(video_path: str) -> list[dict]:
    """
    Run MediaPipe Pose on each frame and return a list of per-frame records.

    Each record looks like:
        {"left":  {"shoulder": (x, y, vis), ...},
         "right": {"shoulder": (x, y, vis), ...}}
    """
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

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
                record = {}
                for side, joints in SIDES.items():
                    record[side] = {}
                    for name, idx in joints.items():
                        lm = landmarks[idx]
                        vis = getattr(lm, "visibility", 1.0)
                        record[side][name] = (lm.x * w, lm.y * h, vis)
                all_frames.append(record)

            frame_idx += 1

    cap.release()
    return all_frames


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
    """
    clean = []
    for rec in frames:
        joints = rec[side]
        if all(joints[j][2] >= MIN_VISIBILITY for j in needed):
            clean.append({j: (joints[j][0], joints[j][1]) for j in needed})
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

    frames = extract_landmarks(video_path)

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
        return res

    depth_good = bottom_elbow <= PU_DEPTH_MAX
    align_good = body_align >= PU_ALIGN_MIN

    strengths, tips = [], []
    (strengths if depth_good else tips).append(_m(language, "pushup", "depth_good" if depth_good else "depth_bad"))
    (strengths if align_good else tips).append(_m(language, "pushup", "align_good" if align_good else "align_bad"))

    overall = _verdict(sum([depth_good, align_good]), 2)
    summary = _m(language, "pushup", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "min_elbow_angle": round(bottom_elbow, 1),
        "avg_body_alignment": round(body_align, 1),
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
        return _unknown(exercise, language, _m(language, "squat", "norep"), len(clean))

    depth_good = bottom_knee <= SQ_DEPTH_MAX
    back_good = avg_lean <= SQ_LEAN_MAX

    strengths, tips = [], []
    (strengths if depth_good else tips).append(_m(language, "squat", "depth_good" if depth_good else "depth_bad"))
    (strengths if back_good else tips).append(_m(language, "squat", "back_good" if back_good else "back_bad"))

    overall = _verdict(sum([depth_good, back_good]), 2)
    summary = _m(language, "squat", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "min_knee_angle": round(bottom_knee, 1),
        "avg_torso_lean": round(avg_lean, 1),
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

    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "avg_body_alignment": round(body_align, 1),
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
        return _unknown(exercise, language, _m(language, "lunge", "norep"), len(clean))

    depth_good = bottom_knee <= LU_DEPTH_MAX
    torso_good = avg_lean <= LU_LEAN_MAX

    strengths, tips = [], []
    (strengths if depth_good else tips).append(_m(language, "lunge", "depth_good" if depth_good else "depth_bad"))
    (strengths if torso_good else tips).append(_m(language, "lunge", "torso_good" if torso_good else "torso_bad"))

    overall = _verdict(sum([depth_good, torso_good]), 2)
    summary = _m(language, "lunge", {"great": "great", "okay": "okay", "needs work": "needs"}[overall])

    return {
        "exercise": exercise,
        "overall": overall,
        "strengths": strengths,
        "tips": tips,
        "summary": summary,
        "frames_analyzed": len(clean),
        "min_knee_angle": round(bottom_knee, 1),
        "avg_torso_lean": round(avg_lean, 1),
    }
