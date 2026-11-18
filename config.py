"""
════════════════════════════════════════════════════════════════════════════════
config.py — Centralised Configuration for Real-Time Surveillance System
════════════════════════════════════════════════════════════════════════════════

Every threshold, path, and constant used by the system is defined here.
No magic numbers are permitted anywhere else in the codebase.
"""

# ── Hardware ──────────────────────────────────────────────────────────────────
NUM_THREADS: int = 14  # Ryzen 7 5700U: 16 logical threads − 2 reserved for OS

# ── Video ─────────────────────────────────────────────────────────────────────
VIDEO_SOURCE = 0 # 0 = webcam | "path/video.mp4" | "rtsp://..." URL
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 640

# ── CPU Frame-Skip ────────────────────────────────────────────────────────────
DETECTION_SKIP: int = 2      # person+bag YOLO runs every 2 frames
WEAPON_SKIP: int = 3         # weapon YOLO runs every 3 frames
SCENE_INFER_EVERY: int = 60  # scene classifier runs every 60 frames

# ── Models ────────────────────────────────────────────────────────────────────
PERSON_BAG_MODEL: str = "yolov8n.pt"
PERSON_BAG_CONF: float = 0.65
BAG_CONF: float = 0.15      # lower threshold specifically for bags to prevent flickering
PERSON_BAG_IMGSZ: int = 416

WEAPON_MODEL: str = "weapon_yolo11s.pt"
WEAPON_CONF: float = 0.60   # raised to reduce false positives
WEAPON_CONFIRM_FRAMES: int = 12 # ~1.5 seconds confirmation time to prevent flicker alarms
WEAPON_IMGSZ: int = 416

PLACES365_MODEL: str = "resnet18_places365.pth.tar"
PLACES365_LABELS: str = "categories_places365.txt"
SCENE_TOP_K: int = 3

# ── SORT Tracker ──────────────────────────────────────────────────────────────
SORT_MAX_AGE: int = 120  # keep lost tracks alive ~4 sec for brief exits/occlusions
SORT_MIN_HITS: int = 2
SORT_IOU: float = 0.3

# ── Behaviour Thresholds ─────────────────────────────────────────────────────
LOITER_SPEED_PX: float = 5.0        # px/frame below which = "stationary" (still)
LOITER_ROAM_RADIUS_PX: float = 80.0 # max displacement from average pos = "pacing"
LOITER_ROAM_WINDOW: int = 60        # frames of position history to check for pacing
LOITER_ZONE_PROXIMITY_PX: float = 120.0  # only detect loitering within this distance of a geofence zone
LOITER_DECAY_RATE: int = 5          # frames subtracted per frame of movement (gradual decay, not hard reset)
LOITER_FRAMES_SUSPICIOUS: int = 300  # ~10 sec → SUSPICIOUS (label on screen)
LOITER_FRAMES_HIGH: int = 600        # ~20 sec → HIGH (alarm triggers)
LOITER_FRAMES_CRITICAL: int = 1800   # ~60 sec near restricted zone → CRITICAL

FAST_SPEED_PX: float = 60.0         # px/frame threshold for fast movement
CROWD_COUNT: int = 4                # unique IDs in frame = crowd

PERSON_PROXIMITY_PX: float = 80.0   # person ↔ weapon = suspicious
PERSON_PERSON_PROX_PX: float = 60.0 # person ↔ person altercation distance

# ── Abandoned Bag State Machine ───────────────────────────────────────────────
BAG_SETTLE_RADIUS_PX: float = 40.0   # centroid drift < this = "Stationary" (increased to ignore bbox jitter)
BAG_SETTLE_SECONDS: float = 3.0      # must be stationary for 3 s to confirm
BAG_OWNER_TOUCH_PX: float = 120.0    # distance to link person as owner (increased for easier association)
BAG_SEPARATION_PX: float = 200.0     # owner walked-away threshold
BAG_ABANDON_SUSPICIOUS: float = 15.0 # seconds alone → SUSPICIOUS
BAG_ABANDON_HIGH: float = 30.0       # seconds alone → HIGH
BAG_ABANDON_CRITICAL: float = 90.0   # seconds alone in lobby/hall → CRITICAL

# ── Geofencing ────────────────────────────────────────────────────────────────
# Polygonal restricted zones as lists of (x, y) pixel vertices.
# Coordinates are frame-relative at FRAME_WIDTH × FRAME_HEIGHT (640 × 480).
# Configure to match actual camera layout. Empty list = no zones.
GEOFENCE_ZONES: list = []

# ── Restricted Scenes ─────────────────────────────────────────────────────────
RESTRICTED_KEYWORDS: set = {
    "server_room", "vault", "restricted", "storage",
    "control_room"
    # NOTE: "office" and "corridor" removed — Places365 classifies most
    # indoor rooms as "office" or "corridor", causing false CRITICAL alerts.
    # Add them back only if your camera monitors a genuinely restricted area.
}

# ── Context-Aware Scene-Weapon Threat Deltas ──────────────────────────────────
# Applied additively when a weapon is present. First match wins.
SCENE_WEAPON_DELTA: dict = {
    "kitchen":    -0.35,
    "public":     +0.35,
    "lobby":      +0.30,
    "corridor":   +0.25,
    "parking":    +0.20,
    "office":     +0.20,
    "restricted": +0.45,
    "hall":       +0.30,
}

# ── Threat Level Thresholds ───────────────────────────────────────────────────
THREAT_SUSPICIOUS: float = 0.25  # yellow UI notification only
THREAT_LOW: float = 0.40         # voice + email
THREAT_HIGH: float = 0.60        # siren + voice + email + screenshot
THREAT_CRITICAL: float = 0.80    # siren + voice + email + screenshot (loudest)

# ── Alert Settings ────────────────────────────────────────────────────────────
ALARM_SOUND: str = "alarm.wav"   # used for HIGH alerts
SIREN_SOUND: str = "siren.wav"   # used for CRITICAL alerts
ALERT_COOLDOWN: float = 6.0     # seconds between same-category alerts
ALERT_DIR: str = "alerts"

# ── Display ───────────────────────────────────────────────────────────────────
SHOW_TRAJECTORIES: bool = True
TRAJECTORY_LEN: int = 25
