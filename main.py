"""
════════════════════════════════════════════════════════════════════════════════
main.py — Real-Time Intelligent Surveillance System
Multi-Condition Threat Assessment with Context-Aware Analysis
════════════════════════════════════════════════════════════════════════════════

Entry point.  Implements nine modules:
    M1+M2  VideoSource          — threaded frame grabber
    M3     ObjectDetector       — dual YOLOv8 inference (person/bag + weapon)
    M4     ObjectTracker        — dual SORT tracker (persons + bags)
    M5     BehaviourAnalyser    — 6-behaviour analysis + abandoned-bag FSM
    M6     SceneClassifier      — ResNet18 + Places365 (background thread)
    M7     ThreatAnalyser       — context-aware multi-condition rule engine
    M8     AlertManager         — cooldown-based audio / email / screenshot alerts
    M9     Display              — HUD, overlays, trajectories, geofence zones
    Main   main()               — orchestration loop
"""

# ══════════════════════════════════════════════════════════════════════════════
# CPU OPTIMISATIONS — must precede all other imports
# ══════════════════════════════════════════════════════════════════════════════
import os

# Auto-cd into the script's own directory so relative model paths resolve
# correctly even when launched from a parent folder.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load .env file (SMTP credentials, etc.) — must be after chdir so .env is found
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

os.environ["OMP_NUM_THREADS"] = "14"
os.environ["MKL_NUM_THREADS"] = "14"
os.environ["OPENBLAS_NUM_THREADS"] = "14"

import torch  # noqa: E402

torch.set_num_threads(14)
torch.set_num_interop_threads(4)

import cv2  # noqa: E402

cv2.setNumThreads(14)

# ══════════════════════════════════════════════════════════════════════════════
# Standard / third-party imports
# ══════════════════════════════════════════════════════════════════════════════
import logging
import math
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from ultralytics import YOLO

import config
import sendmail
from sort import Sort

# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_fmt, handlers=[
    logging.FileHandler("surveillance.log", encoding="utf-8"),
    logging.StreamHandler(),
])
log = logging.getLogger("surveillance")

# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BagState:
    """Internal state for a single tracked bag in the abandoned-bag FSM."""

    bag_id: int
    state: str = "Carried"  # "Carried" | "Stationary" | "Monitoring" | "Abandoned"
    settle_start: float | None = None
    owner_id: int | None = None
    abandon_start: float | None = None
    last_centroid: tuple[int, int] = (0, 0)
    abandon_level: str = ""  # "" | "SUSPICIOUS" | "HIGH" | "CRITICAL"
    last_seen: float = 0.0


@dataclass
class AbandonedBag:
    """Output record for a bag that has entered an abandoned state."""

    bag_id: int
    centroid: tuple[int, int]
    duration: float  # seconds since abandonment started
    level: str  # "SUSPICIOUS" | "HIGH" | "CRITICAL"


@dataclass
class BehaviourFlags:
    """Aggregated behavioural analysis results for a single frame."""

    loitering: list[int] = field(default_factory=list)
    loitering_level: dict[int, str] = field(default_factory=dict)
    crowd: bool = False
    fast_move: list[int] = field(default_factory=list)
    confirmed_weapons: list[str] = field(default_factory=list)

    abandoned_bags: list[AbandonedBag] = field(default_factory=list)
    geofence_breach: list[int] = field(default_factory=list)
    speeds: dict[int, float] = field(default_factory=dict)


@dataclass
class ThreatResult:
    """Output of the context-aware threat analysis rule engine."""

    level: str = "SAFE"  # "SAFE" | "SUSPICIOUS" | "HIGH" | "CRITICAL"
    score: float = 0.0  # 0.0 – 1.0
    matched_rule: str = ""
    reasons: list[str] = field(default_factory=list)
    alert_type: str = "none"  # "none" | "log_only" | "voice_email" | "full_siren"


# ══════════════════════════════════════════════════════════════════════════════
# M1 + M2 — VideoSource
# ══════════════════════════════════════════════════════════════════════════════


class VideoSource:
    """Threaded video capture from webcam, file, or RTSP stream.

    A background daemon thread continuously grabs frames so the main loop
    never blocks on I/O.  ``read()`` returns the latest frame instantly.
    """

    def __init__(self, source: int | str, width: int, height: int) -> None:
        self._source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # For file-based sources, get native FPS for throttled playback
        self._is_file = isinstance(source, str) and not str(source).startswith("rtsp")
        self._frame_delay = 0.0
        if self._is_file:
            native_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if native_fps and native_fps > 0:
                self._frame_delay = 1.0 / native_fps
            else:
                self._frame_delay = 1.0 / 30.0  # default to 30fps

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        log.info("VideoSource started — source=%s  %dx%d", source, width, height)

    # ── background reader ────────────────────────────────────────────────
    def _reader(self) -> None:
        """Continuously grab frames into ``self._frame``."""
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # For video files: loop back to start
                if self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
            # Throttle video file playback to real-time speed
            if self._is_file and self._frame_delay > 0:
                time.sleep(self._frame_delay)

    # ── public API ───────────────────────────────────────────────────────
    def read(self) -> np.ndarray | None:
        """Return a copy of the latest frame, or ``None`` if unavailable."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_alive(self) -> bool:
        """Return ``True`` while the reader thread is active."""
        return not self._stop.is_set()

    def release(self) -> None:
        """Signal the reader thread to stop and release the capture device."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
        log.info("VideoSource released.")


# ══════════════════════════════════════════════════════════════════════════════
# M3 — ObjectDetector
# ══════════════════════════════════════════════════════════════════════════════

# COCO class names we care about from the general model
_PERSON_BAG_NAMES = {"person", "backpack", "handbag", "suitcase"}

# Weapon model class mapping (YOLOv11s — 2 classes)
_WEAPON_NAMES = {0: "knife", 1: "pistol"}


class ObjectDetector:
    """Dual YOLO detector — person/bag (COCO YOLOv8n) + weapon (custom YOLOv11s, 2-class: knife, pistol).

    Frame-skip caching avoids running inference on every frame.  Between
    inference frames the SORT tracker fills gaps via Kalman prediction.
    """

    def __init__(self) -> None:
        try:
            self._person_bag_model = YOLO(config.PERSON_BAG_MODEL)
            log.info("Loaded person/bag model: %s", config.PERSON_BAG_MODEL)
        except Exception:
            log.exception("Failed to load person/bag model: %s", config.PERSON_BAG_MODEL)
            raise

        try:
            self._weapon_model = YOLO(config.WEAPON_MODEL)
            log.info("Loaded weapon model: %s", config.WEAPON_MODEL)
        except Exception:
            log.exception("Failed to load weapon model: %s", config.WEAPON_MODEL)
            raise

        self._pb_cache: list[dict[str, Any]] = []
        self._wp_cache: list[dict[str, Any]] = []

    # ── public API ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, frame_n: int) -> list[dict[str, Any]]:
        """Run detection (with frame-skip caching) and return a merged list.

        Each returned dict: ``{x1, y1, x2, y2, conf, label, category}``.
        """
        frame_area = frame.shape[0] * frame.shape[1]

        # Person / bag inference
        if frame_n % config.DETECTION_SKIP == 0:
            self._pb_cache = self._run_person_bag(frame, frame_area)

        # Weapon inference
        if frame_n % config.WEAPON_SKIP == 0:
            self._wp_cache = self._run_weapon(frame, frame_area)

        return self._pb_cache + self._wp_cache

    # ── internal helpers ─────────────────────────────────────────────────
    @torch.inference_mode()
    def _run_person_bag(self, frame: np.ndarray, frame_area: int) -> list[dict[str, Any]]:
        """Infer persons and bags with the COCO-pretrained YOLOv8n."""
        results = self._person_bag_model(
            frame,
            conf=config.BAG_CONF,  # run model at lower threshold to catch bags
            imgsz=config.PERSON_BAG_IMGSZ,
            device="cpu",
            verbose=False,
        )
        dets: list[dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                label = r.names.get(cls_id, "")
                if label not in _PERSON_BAG_NAMES:
                    continue
                
                conf = float(boxes.conf[i].item())
                if label == "person" and conf < config.PERSON_BAG_CONF:
                    continue  # filter persons back to normal confidence

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < 0.005 * frame_area:
                    continue  # false-positive filter
                category = "person" if label == "person" else "bag"
                dets.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf": float(boxes.conf[i].item()),
                    "label": label,
                    "category": category,
                })
        return dets

    @torch.inference_mode()
    def _run_weapon(self, frame: np.ndarray, frame_area: int) -> list[dict[str, Any]]:
        """Infer weapons with the custom 2-class YOLOv11s model (knife, pistol)."""
        results = self._weapon_model(
            frame,
            conf=config.WEAPON_CONF,
            imgsz=config.WEAPON_IMGSZ,
            device="cpu",
            verbose=False,
        )
        dets: list[dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                label = _WEAPON_NAMES.get(cls_id, f"weapon_{cls_id}")
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < 0.005 * frame_area:
                    continue
                dets.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf": float(boxes.conf[i].item()),
                    "label": label,
                    "category": "weapon",
                })
        return dets


# ══════════════════════════════════════════════════════════════════════════════
# M4 — ObjectTracker (SORT)
# ══════════════════════════════════════════════════════════════════════════════


class ObjectTracker:
    """Dual SORT tracker — separate instances for persons and bags.

    Maintains per-ID centroid history for trajectory drawing and speed
    computation.
    """

    def __init__(self) -> None:
        self.person_tracker = Sort(
            max_age=config.SORT_MAX_AGE,
            min_hits=config.SORT_MIN_HITS,
            iou_threshold=config.SORT_IOU,
        )
        self.bag_tracker = Sort(
            max_age=config.SORT_MAX_AGE,
            min_hits=config.SORT_MIN_HITS,
            iou_threshold=config.SORT_IOU,
        )
        self.person_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.TRAJECTORY_LEN)
        )
        self.bag_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.TRAJECTORY_LEN)
        )
        log.info("ObjectTracker initialised (dual SORT).")

    def update(
        self, detections: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Feed detections into SORT and return tracked persons and bags.

        Returns
        -------
        tracked_persons : list[dict]
            Each: ``{x1, y1, x2, y2, cx, cy, track_id, label, category, conf}``
        tracked_bags : list[dict]
            Same schema as ``tracked_persons``.
        """
        person_dets = [d for d in detections if d["category"] == "person"]
        bag_dets = [d for d in detections if d["category"] == "bag"]

        tracked_persons = self._track(person_dets, self.person_tracker,
                                      self.person_history, "person")
        tracked_bags = self._track(bag_dets, self.bag_tracker,
                                   self.bag_history, "bag")
        return tracked_persons, tracked_bags

    # ── internal ─────────────────────────────────────────────────────────
    @staticmethod
    def _track(
        dets: list[dict[str, Any]],
        tracker: Sort,
        history: dict[int, deque],
        default_category: str,
    ) -> list[dict[str, Any]]:
        """Run a single SORT instance and return enriched track dicts."""
        if len(dets) == 0:
            sort_input = np.empty((0, 5))
        else:
            sort_input = np.array(
                [[d["x1"], d["y1"], d["x2"], d["y2"], d["conf"]] for d in dets]
            )

        tracks = tracker.update(sort_input)  # Nx5: x1,y1,x2,y2,id
        result: list[dict[str, Any]] = []
        for trk in tracks:
            x1, y1, x2, y2, tid = trk[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            tid = int(tid)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            history[tid].append((cx, cy))

            # Find the best matching detection for label info
            best_label = default_category
            best_conf = 0.0
            for d in dets:
                iou = _box_iou(
                    (x1, y1, x2, y2),
                    (d["x1"], d["y1"], d["x2"], d["y2"]),
                )
                if iou > 0.3 and d["conf"] > best_conf:
                    best_label = d["label"]
                    best_conf = d["conf"]

            result.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
                "track_id": tid,
                "label": best_label,
                "category": default_category,
                "conf": best_conf if best_conf > 0 else 0.5,
            })
        return result


def _box_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# M5 — BehaviourAnalyser
# ══════════════════════════════════════════════════════════════════════════════


class BehaviourAnalyser:
    """Core intelligence module — analyses six behavioural indicators.

    Behaviours detected:
        1. Loitering (3-tier escalation)
        2. Crowd formation
        3. Fast movement / fleeing
        4. Abandoned bag (4-phase state machine)
        5. Suspicious proximity (person↔weapon, person↔person)
        6. Geofence breach
    """

    def __init__(self) -> None:
        # Loitering state
        self.stationary_count: dict[int, int] = defaultdict(int)
        self.last_pos: dict[int, tuple[int, int]] = {}
        # Position history for pacing detection (track_id -> deque of (cx, cy))
        self.pos_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.LOITER_ROAM_WINDOW)
        )

        # Abandoned bag FSM
        self.bag_states: dict[int, BagState] = {}

        # Geofence — refreshed each frame
        self._geofence_breached_ids: set[int] = set()
        
        # Weapon confirmation state
        self.weapon_counters: dict[str, int] = {"pistol": 0, "knife": 0}

        log.info("BehaviourAnalyser initialised.")

    # ── helpers ──────────────────────────────────────────────────────────
    def _is_near_geofence_zone(self, cx: int, cy: int) -> bool:
        """Return True if (cx, cy) is within LOITER_ZONE_PROXIMITY_PX of any
        geofence zone polygon boundary or interior."""
        if not config.GEOFENCE_ZONES:
            return False
        for zone in config.GEOFENCE_ZONES:
            polygon = np.array(zone["polygon"], dtype=np.int32)
            # pointPolygonTest returns: +ve if inside, 0 on edge,
            # -ve = distance to nearest edge (when measureDist=True)
            dist = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), True)
            # dist >= 0 means inside the zone, or
            # abs(dist) <= proximity means near the zone boundary
            if dist >= 0 or abs(dist) <= config.LOITER_ZONE_PROXIMITY_PX:
                return True
        return False

    def _is_pacing(self, tid: int) -> bool:
        """Return True if the person is pacing — moving but staying within a
        confined area (their position history bounding-box is small)."""
        hist = self.pos_history.get(tid)
        if not hist or len(hist) < 10:
            return False
        xs = [pt[0] for pt in hist]
        ys = [pt[1] for pt in hist]
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        # If the person's total movement spread is within the roam radius,
        # they're pacing in a confined area (not genuinely travelling)
        return max(spread_x, spread_y) <= config.LOITER_ROAM_RADIUS_PX

    # ── public API ───────────────────────────────────────────────────────
    def analyse(
        self,
        tracked_persons: list[dict[str, Any]],
        tracked_bags: list[dict[str, Any]],
        all_detections: list[dict[str, Any]],
        fps: float,
        frame: np.ndarray,
    ) -> BehaviourFlags:
        """Run all six behaviour checks and return aggregated flags."""
        flags = BehaviourFlags()
        active_person_ids: set[int] = set()

        # ── Per-person analysis ──────────────────────────────────────────
        for p in tracked_persons:
            tid = p["track_id"]
            cx, cy = p["cx"], p["cy"]
            active_person_ids.add(tid)

            # Speed computation
            speed = 0.0
            if tid in self.last_pos:
                lx, ly = self.last_pos[tid]
                speed = math.hypot(cx - lx, cy - ly)
            flags.speeds[tid] = speed
            self.last_pos[tid] = (cx, cy)

            # Update position history for pacing detection
            self.pos_history[tid].append((cx, cy))

            # ── Behaviour 1: Loitering (only when geofence zones exist) ───
            # Loitering is detected anywhere in the frame, but ONLY when at
            # least one restricted zone has been defined.  A person is
            # "loitering" if they are standing still OR pacing in a confined area.
            zones_exist = len(config.GEOFENCE_ZONES) > 0
            is_still = speed < config.LOITER_SPEED_PX
            is_pacing = self._is_pacing(tid)

            if zones_exist and (is_still or is_pacing):
                self.stationary_count[tid] += 1
            else:
                # Gradual decay instead of hard reset — brief movements
                # don't erase accumulated loitering history
                self.stationary_count[tid] = max(
                    0, self.stationary_count[tid] - config.LOITER_DECAY_RATE
                )

            sc = self.stationary_count[tid]
            if sc >= config.LOITER_FRAMES_SUSPICIOUS:
                flags.loitering.append(tid)
                if sc >= config.LOITER_FRAMES_CRITICAL and self.in_restricted_zone(tid):
                    flags.loitering_level[tid] = "CRITICAL"
                elif sc >= config.LOITER_FRAMES_HIGH:
                    flags.loitering_level[tid] = "HIGH"
                else:
                    flags.loitering_level[tid] = "SUSPICIOUS"

            # ── Behaviour 3: Fast movement / fleeing ─────────────────────
            if speed > config.FAST_SPEED_PX:
                flags.fast_move.append(tid)

        # ── Behaviour 2: Crowd formation ─────────────────────────────────
        if len(tracked_persons) >= config.CROWD_COUNT:
            flags.crowd = True

        # ── Behaviour 4: Abandoned bag state machine ─────────────────────
        self._analyse_abandoned_bags(tracked_bags, tracked_persons, flags)

        weapon_dets = [d for d in all_detections if d["category"] == "weapon"]
        
        # ── Behaviour 7: Weapon Confirmation ─────────────────────────────
        current_weapons = set(w["label"] for w in weapon_dets)
        for w_type in ("pistol", "knife"):
            if w_type in current_weapons:
                # Increment counter
                self.weapon_counters[w_type] += 1
                if self.weapon_counters[w_type] > config.WEAPON_CONFIRM_FRAMES:
                    self.weapon_counters[w_type] = config.WEAPON_CONFIRM_FRAMES
            else:
                # Decay counter
                self.weapon_counters[w_type] = max(0, self.weapon_counters[w_type] - 1)
            
            # If threshold reached, it's confirmed
            if self.weapon_counters[w_type] >= config.WEAPON_CONFIRM_FRAMES:
                flags.confirmed_weapons.append(w_type)

        # (Behaviour 5a: Person ↔ Weapon proximity — removed)



        # ── Behaviour 6: Geofence breach ────────────────────────────────
        self._geofence_breached_ids.clear()
        for zone in config.GEOFENCE_ZONES:
            polygon = np.array(zone["polygon"], dtype=np.int32)
            for p in tracked_persons:
                result = cv2.pointPolygonTest(
                    polygon, (float(p["cx"]), float(p["cy"])), False
                )
                if result >= 0:  # inside or on boundary
                    flags.geofence_breach.append(p["track_id"])
                    self._geofence_breached_ids.add(p["track_id"])

        # ── Cleanup stale loitering state ────────────────────────────────
        stale = [k for k in self.stationary_count if k not in active_person_ids]
        for k in stale:
            del self.stationary_count[k]
            self.last_pos.pop(k, None)
            self.pos_history.pop(k, None)



        return flags

    # ── helper ───────────────────────────────────────────────────────────
    def in_restricted_zone(self, tid: int) -> bool:
        """Return ``True`` if *tid* is currently inside a geofenced zone."""
        return tid in self._geofence_breached_ids

    # ── Abandoned bag FSM ────────────────────────────────────────────────
    def _analyse_abandoned_bags(
        self,
        tracked_bags: list[dict[str, Any]],
        tracked_persons: list[dict[str, Any]],
        flags: BehaviourFlags,
    ) -> None:
        """Run the 4-phase abandoned-bag state machine for every tracked bag."""
        active_bag_ids: set[int] = set()

        for bag in tracked_bags:
            bid = bag["track_id"]
            bcx, bcy = bag["cx"], bag["cy"]
            active_bag_ids.add(bid)

            # Initialise state if new bag
            if bid not in self.bag_states:
                self.bag_states[bid] = BagState(
                    bag_id=bid,
                    last_centroid=(bcx, bcy),
                    last_seen=time.time()
                )

            bs = self.bag_states[bid]
            now = time.time()
            bs.last_seen = now

            # ── PHASE 1: Stationary validation ───────────────────────────
            drift = math.hypot(bcx - bs.last_centroid[0], bcy - bs.last_centroid[1])

            if bs.state == "Carried":
                if drift < config.BAG_SETTLE_RADIUS_PX:
                    if bs.settle_start is None:
                        bs.settle_start = now
                    elif now - bs.settle_start >= config.BAG_SETTLE_SECONDS:
                        bs.state = "Stationary"
                        # Fall through to Phase 2
                else:
                    bs.settle_start = None  # bag is moving, reset

            bs.last_centroid = (bcx, bcy)

            # ── PHASE 2 & 3: Supervision & Separation ────────────────────
            if bs.state in ("Stationary", "Monitoring", "Abandoned"):
                bs.state = "Monitoring"  # Ensure state moves forward
                
                # Find the closest person to the bag
                min_dist = float("inf")
                closest_pid: int | None = None
                for p in tracked_persons:
                    d = math.hypot(p["cx"] - bcx, p["cy"] - bcy)
                    if d < min_dist:
                        min_dist = d
                        closest_pid = p["track_id"]

                # If ANY person is within the separation radius, the bag is supervised!
                if closest_pid is not None and min_dist <= config.BAG_SEPARATION_PX:
                    bs.owner_id = closest_pid
                    bs.abandon_start = None  # Reset timer! Person is nearby.
                    bs.abandon_level = ""
                else:
                    # No person is nearby -> bag is unsupervised.
                    if bs.abandon_start is None:
                        bs.abandon_start = now

            # ── PHASE 4: Time threshold ──────────────────────────────────
            if bs.abandon_start is not None:
                elapsed = now - bs.abandon_start
                if elapsed >= config.BAG_ABANDON_CRITICAL:
                    bs.abandon_level = "CRITICAL"
                    bs.state = "Abandoned"
                    flags.abandoned_bags.append(
                        AbandonedBag(bid, (bcx, bcy), elapsed, "CRITICAL")
                    )
                elif elapsed >= config.BAG_ABANDON_HIGH:
                    bs.abandon_level = "HIGH"
                    bs.state = "Abandoned"
                    flags.abandoned_bags.append(
                        AbandonedBag(bid, (bcx, bcy), elapsed, "HIGH")
                    )
                elif elapsed >= config.BAG_ABANDON_SUSPICIOUS:
                    bs.abandon_level = "SUSPICIOUS"
                    flags.abandoned_bags.append(
                        AbandonedBag(bid, (bcx, bcy), elapsed, "SUSPICIOUS")
                    )

        # Cleanup states for bags no longer tracked (with 5-second grace period for flickers)
        now = time.time()
        stale = [
            k for k, bs in self.bag_states.items()
            if k not in active_bag_ids and (now - bs.last_seen) > 5.0
        ]
        for k in stale:
            del self.bag_states[k]


# ══════════════════════════════════════════════════════════════════════════════
# M6 — SceneClassifier (ResNet18 + Places365)
# ══════════════════════════════════════════════════════════════════════════════


class SceneClassifier:
    """Background-threaded scene classifier using ResNet18 + Places365.

    Inference runs in a dedicated daemon thread.  ``submit()`` drops frames
    if the worker is busy (non-blocking).  ``result`` returns the cached
    prediction instantly.
    """

    def __init__(self) -> None:
        # ── Load labels ──────────────────────────────────────────────────
        labels_path = Path(config.PLACES365_LABELS)
        if not labels_path.is_file():
            log.warning("Places365 labels file not found: %s — scene classification disabled.", labels_path)
            self._labels: list[str] = []
            self._model: nn.Module | None = None
            self._last_result: list[tuple[str, float]] = [("unknown", 0.0)]
            self._queue: queue.Queue = queue.Queue(maxsize=1)
            return

        with open(labels_path, "r", encoding="utf-8") as f:
            self._labels = []
            for line in f:
                line = line.strip()
                if line:
                    # Strip leading "/" and take the first token
                    token = line.lstrip("/").split(" ")[0].split("/")[-1]
                    self._labels.append(token)

        # ── Build model ──────────────────────────────────────────────────
        try:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(512, 365)

            ckpt_path = Path(config.PLACES365_MODEL)
            if not ckpt_path.is_file():
                log.warning("Places365 weights not found: %s — scene classification disabled.", ckpt_path)
                self._model = None
                self._last_result = [("unknown", 0.0)]
                self._queue = queue.Queue(maxsize=1)
                return

            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
            # Handle .pth.tar format (checkpoint dict with 'state_dict' key)
            # vs plain .pth format (raw state_dict)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            # Strip "module." prefix if present (DataParallel artifact)
            cleaned: dict[str, Any] = {}
            for k, v in state_dict.items():
                key = k.replace("module.", "", 1) if k.startswith("module.") else k
                cleaned[key] = v
            model.load_state_dict(cleaned)
            model.eval()

            # NOTE: torch.compile(backend="inductor") is skipped because it
            # requires a C++ compiler (MSVC cl.exe on Windows) which may not be
            # installed.  Eager mode is sufficient for ResNet18 on CPU.

            self._model = model
            log.info("SceneClassifier loaded (%d labels).", len(self._labels))

        except Exception:
            log.exception("Failed to load SceneClassifier model.")
            self._model = None

        # ── Preprocessing pipeline ───────────────────────────────────────
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # ── Thread-safe result cache ─────────────────────────────────────
        self._last_result: list[tuple[str, float]] = [("unknown", 0.0)]
        self._result_lock = threading.Lock()

        # ── Submission queue + worker ────────────────────────────────────
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    # ── background worker ────────────────────────────────────────────────
    def _worker(self) -> None:
        """Continuously consume frames from the queue and run inference."""
        while True:
            frame = self._queue.get()  # blocks
            if self._model is None:
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self._transform(rgb).unsqueeze(0)  # (1,3,224,224)

                with torch.inference_mode():
                    logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, config.SCENE_TOP_K)

                result: list[tuple[str, float]] = []
                for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                    label = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
                    result.append((label, prob))

                with self._result_lock:
                    self._last_result = result
            except Exception:
                log.exception("SceneClassifier inference error.")

    # ── public API ───────────────────────────────────────────────────────
    def submit(self, frame: np.ndarray) -> None:
        """Submit a frame for classification.  Drops if worker is busy."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass  # drop silently — non-blocking

    @property
    def result(self) -> list[tuple[str, float]]:
        """Return the latest classification result (thread-safe)."""
        with self._result_lock:
            return list(self._last_result)

    def top_label(self) -> str:
        """Return the top-1 scene label string."""
        r = self.result
        return r[0][0] if r else "unknown"

    def is_restricted(self) -> bool:
        """Return ``True`` if the top label matches any restricted keyword."""
        label = self.top_label().lower()
        return any(kw in label for kw in config.RESTRICTED_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# M7 — ThreatAnalyser (Context-Aware Rule Engine)
# ══════════════════════════════════════════════════════════════════════════════


class ThreatAnalyser:
    """Multi-condition context-aware threat assessment engine.

    Evaluates every combination of (object + behaviour + scene) against an
    ordered rule set.  CRITICAL rules are checked first, then HIGH,
    SUSPICIOUS, SAFE, and finally a weighted DEFAULT.
    """

    def __init__(self) -> None:
        log.info("ThreatAnalyser initialised.")

    # ── public API ───────────────────────────────────────────────────────
    def analyse(
        self,
        detections: list[dict[str, Any]],
        behaviour: BehaviourFlags,
        scene_label: str,
        is_restricted: bool,
        tracked_persons: list[dict[str, Any]],
        tracked_bags: list[dict[str, Any]],
    ) -> ThreatResult:
        """Evaluate all rules and return the highest-priority match."""
        scene = scene_label.lower()
        weapon_labels = set(behaviour.confirmed_weapons)
        weapon_present = len(weapon_labels) > 0
        person_count = len(tracked_persons)
        bag_present = len(tracked_bags) > 0

        reasons: list[str] = []
        if weapon_present:
            reasons.append(f"Confirmed weapons detected: {weapon_labels}")
        if person_count > 0:
            reasons.append(f"Persons: {person_count}")
        if behaviour.crowd:
            reasons.append("Crowd formation")
        if behaviour.loitering:
            reasons.append(f"Loitering IDs: {behaviour.loitering}")
        if behaviour.fast_move:
            reasons.append(f"Fast movement IDs: {behaviour.fast_move}")

        if behaviour.abandoned_bags:
            reasons.append(f"Abandoned bags: {len(behaviour.abandoned_bags)}")
        if behaviour.geofence_breach:
            reasons.append(f"Geofence breach IDs: {behaviour.geofence_breach}")

        knife_present = "knife" in weapon_labels
        pistol_present = "pistol" in weapon_labels

        # Helper: check if scene contains any of the given keywords
        def scene_has(*keywords: str) -> bool:
            return any(kw in scene for kw in keywords)

        # Fast-move IDs set
        fast_ids = set(behaviour.fast_move)

        # ═════════════════════════════════════════════════════════════════
        # SAFE RULES — checked FIRST to prevent false alarms
        # ═════════════════════════════════════════════════════════════════

        # (SAFE1 removed — knife is now a universally restricted weapon)

        # (SAFE2 removed — no maintenance tool classes in model)

        # SAFE3: person + bag + normal movement + lobby/hall, no weapon
        if bag_present and not behaviour.fast_move and not weapon_present and scene_has("lobby", "hall"):
            return ThreatResult("SAFE", 0.05,
                                "SAFE: Normal visitor/traveller with bag",
                                reasons, "none")

        # (SAFE4 removed — crowd rules simplified to C6 + H3 only)

        # (SAFE5 removed — person proximity feature removed)

        # SAFE6: knife + kitchen = normal cooking activity (context-aware)
        if "knife" in weapon_labels and "pistol" not in weapon_labels and scene_has("kitchen"):
            return ThreatResult("SAFE", 0.05,
                                "SAFE: Knife detected in kitchen — normal cooking activity",
                                reasons, "none")

        # ═════════════════════════════════════════════════════════════════
        # CRITICAL RULES  (score = 1.0, alert = full_siren)
        # ═════════════════════════════════════════════════════════════════

        # C1: Weapon (Pistol or Knife) — universally restricted
        if weapon_present:
            w_str = " and ".join(weapon_labels)
            return ThreatResult("CRITICAL", 1.0,
                                f"CRITICAL: Weapon ({w_str}) confirmed — universally restricted",
                                reasons, "full_siren")

        # (C2: weapon proximity — removed)

        # C4: geofence breach or restricted zone entry
        if behaviour.geofence_breach or (is_restricted and person_count > 0):
            return ThreatResult("CRITICAL", 1.0,
                                "CRITICAL: Unauthorized entry into restricted zone",
                                reasons, "full_siren")

        # C5: abandoned bag CRITICAL + public venue
        if any(ab.level == "CRITICAL" for ab in behaviour.abandoned_bags) and scene_has("lobby", "hall", "public"):
            return ThreatResult("CRITICAL", 1.0,
                                "CRITICAL: Abandoned bag in high-risk public area — IED protocol",
                                reasons, "full_siren")

        # C6: crowd + mass scattering (≥3 fast movers)
        if behaviour.crowd and len(behaviour.fast_move) >= 3:
            return ThreatResult("CRITICAL", 1.0,
                                "CRITICAL: Crowd scattering detected — mass panic or active threat",
                                reasons, "full_siren")



        # ═════════════════════════════════════════════════════════════════
        # HIGH RULES  (score = 0.75, alert = voice_email)
        # ═════════════════════════════════════════════════════════════════



        # H2: person proximity + ≥2 IDs in both proximity and fast_move
        # (H2 removed — person proximity feature removed)

        # H3: crowd near restricted zone
        if behaviour.crowd and is_restricted:
            return ThreatResult("HIGH", 0.75,
                                "HIGH: Unauthorized crowd gathering near restricted area",
                                reasons, "voice_email")

        # H4: abandoned bag HIGH+ in parking/corridor
        if any(ab.level in ("HIGH", "CRITICAL") for ab in behaviour.abandoned_bags) and scene_has("parking", "corridor"):
            return ThreatResult("HIGH", 0.75,
                                "HIGH: Suspicious package in transit zone",
                                reasons, "voice_email")

        # H5: loitering HIGH/CRITICAL near restricted zone
        for tid in behaviour.loitering:
            lvl = behaviour.loitering_level.get(tid, "SUSPICIOUS")
            if lvl in ("HIGH", "CRITICAL") and (is_restricted or behaviour.geofence_breach):
                return ThreatResult("HIGH", 0.75,
                                    "HIGH: Person casing/scouting restricted area entrance",
                                    reasons, "voice_email")

        # (H6 removed — no wrench/hammer classes in model)

        # ═════════════════════════════════════════════════════════════════
        # SUSPICIOUS RULES  (score = 0.45, alert = log_only)
        # ═════════════════════════════════════════════════════════════════

        # S1: loitering SUSPICIOUS + parking
        if any(behaviour.loitering_level.get(t) == "SUSPICIOUS" for t in behaviour.loitering) and scene_has("parking"):
            return ThreatResult("SUSPICIOUS", 0.45,
                                "SUSPICIOUS: Person loitering in parking area — possible car theft scout",
                                reasons, "log_only")

        # S2: loitering + office/corridor
        if behaviour.loitering and scene_has("office", "corridor"):
            return ThreatResult("SUSPICIOUS", 0.45,
                                "SUSPICIOUS: Person lingering in office/corridor",
                                reasons, "log_only")



        # S4: abandoned bag SUSPICIOUS + office
        if any(ab.level == "SUSPICIOUS" for ab in behaviour.abandoned_bags) and scene_has("office"):
            return ThreatResult("SUSPICIOUS", 0.45,
                                "SUSPICIOUS: Unattended bag in office — likely forgotten",
                                reasons, "log_only")

        # (S5 removed — crowd rules simplified to C6 + H3 only)

        # ═════════════════════════════════════════════════════════════════
        # DEFAULT — weighted score from individual signals
        # ═════════════════════════════════════════════════════════════════

        score = 0.0
        if weapon_present:
            score += 0.50
            # Apply scene-weapon delta (first keyword match)
            for kw, delta in config.SCENE_WEAPON_DELTA.items():
                if kw in scene:
                    score += delta
                    break
        if behaviour.loitering:
            score += 0.20


        if behaviour.abandoned_bags:
            high_risk_scenes = ("railway", "subway", "airport", "station", "terminal", "hospital","dorm_room","staircase")
            if any(kw in scene.lower() for kw in high_risk_scenes):
                score += 0.35
        if behaviour.geofence_breach or (is_restricted and person_count > 0):
            score += 0.45

        score = min(score, 1.0)

        # Derive level from thresholds
        if score >= config.THREAT_CRITICAL:
            level, alert = "CRITICAL", "full_siren"
        elif score >= config.THREAT_HIGH:
            level, alert = "HIGH", "voice_email"
        elif score >= config.THREAT_SUSPICIOUS:
            level, alert = "SUSPICIOUS", "log_only"
        else:
            level, alert = "SAFE", "none"

        matched = f"DEFAULT: Weighted threat score {score:.2f}"
        return ThreatResult(level, score, matched, reasons, alert)

    # ── static helpers ───────────────────────────────────────────────────
    @staticmethod
    def color(level: str) -> tuple[int, int, int]:
        """Return a BGR colour tuple for the given threat level."""
        return {
            "CRITICAL":   (0, 0, 255),
            "HIGH":       (0, 140, 255),
            "SUSPICIOUS": (0, 220, 255),
            "SAFE":       (0, 200, 0),
        }.get(level, (200, 200, 200))


# ══════════════════════════════════════════════════════════════════════════════
# M8 — AlertManager
# ══════════════════════════════════════════════════════════════════════════════


class AlertManager:
    """Daemon-threaded alert dispatcher — audio, email, and screenshot.

    All heavy I/O (sound playback, TTS, SMTP) runs on daemon threads so the
    main loop is never blocked.  Per-category cooldowns prevent alert spam.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._tts_lock = threading.Lock()
        Path(config.ALERT_DIR).mkdir(exist_ok=True)

        # Initialise pygame mixer
        try:
            import pygame
            pygame.mixer.init()
            self._mixer_ok = True
            log.info("Pygame mixer initialised.")
        except Exception:
            log.warning("Pygame mixer init failed — audio alerts disabled.")
            self._mixer_ok = False

        log.info("AlertManager initialised — alert dir: %s", config.ALERT_DIR)

    # ── public API ───────────────────────────────────────────────────────
    def trigger(
        self,
        category: str,
        message: str,
        threat_level: str,
        frame: np.ndarray | None = None,
    ) -> None:
        """Fire an alert if the cooldown for *category* has elapsed."""
        now = time.time()
        if now - self._last.get(category, 0.0) < config.ALERT_COOLDOWN:
            return  # cooldown active
        self._last[category] = now

        # Save screenshot (synchronous — ensures file exists before email sends)
        img_path = ""
        if frame is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            img_name = f"{category}_{ts}.jpg"
            img_path = str(Path(config.ALERT_DIR) / img_name)
            self._save_screenshot(frame.copy(), img_path)

        # Audio thread
        threading.Thread(
            target=self._audio,
            args=(message, threat_level),
            daemon=True,
        ).start()

        # Email thread — only for weapon detection and geofence breach
        email_categories = ("critical_threat",)  # covers weapon + geofence from overall threat rules
        geofence_match = category == "geofence"
        weapon_match = "weapon" in message.lower() or "knife" in message.lower() or "pistol" in message.lower()
        if threat_level in ("HIGH", "CRITICAL") and (geofence_match or weapon_match or category in email_categories):
            threading.Thread(
                target=sendmail.sendalert,
                args=(img_path, message, threat_level),
                daemon=True,
            ).start()

        log.info("[%s] Alert triggered — %s: %s", threat_level, category, message)

    def dispatch(
        self,
        threat_result: ThreatResult,
        behaviour: BehaviourFlags,
        is_restricted: bool,
        scene_label: str,
        frame: np.ndarray,
    ) -> None:
        """Dispatch alerts based on the overall threat result and individual behaviours."""
        # ── Overall threat ───────────────────────────────────────────────
        if threat_result.alert_type == "full_siren":
            self.trigger("critical_threat", threat_result.matched_rule, "CRITICAL", frame)
        elif threat_result.alert_type == "voice_email":
            self.trigger("high_threat", threat_result.matched_rule, "HIGH", frame)
        elif threat_result.alert_type == "log_only":
            log.warning("[SUSPICIOUS] %s", threat_result.matched_rule)

        # ── Individual behaviour alerts ──────────────────────────────────
        for tid in behaviour.loitering:
            lvl = behaviour.loitering_level.get(tid, "SUSPICIOUS")
            if lvl in ("HIGH", "CRITICAL"):
                self.trigger(
                    f"loitering_{tid}",
                    f"Loitering detected ID {tid}",
                    lvl, frame,
                )

        # (Standalone crowd alert removed — crowd only alerts via combined rules: mass panic, crowd+restricted zone)

        # Helper: check if scene contains any of the given keywords
        def scene_has(*keywords: str) -> bool:
            return any(kw in scene_label.lower() for kw in keywords)

        # High-risk scenes where abandoned bags trigger audio alerts
        high_risk_scenes = ("railway", "subway", "airport", "station", "terminal", "hospital","dorm_room","staircase")

        for ab in behaviour.abandoned_bags:
            if ab.level in ("HIGH", "CRITICAL"):
                if scene_has(*high_risk_scenes):
                    self.trigger(
                        f"abandoned_{ab.bag_id}",
                        f"Abandoned bag {ab.bag_id} — {ab.duration:.0f}s",
                        ab.level, frame,
                    )
                else:
                    # Just log it quietly if not in a high-risk scene
                    log.warning("[SUSPICIOUS] Abandoned bag %d reached %ds, but scene (%s) is not high-risk.",
                                ab.bag_id, ab.duration, scene_label)

        if behaviour.geofence_breach:
            self.trigger("geofence", "Unauthorized zone entry detected", "CRITICAL", frame)



    # ── internal helpers ─────────────────────────────────────────────────
    @staticmethod
    def _save_screenshot(frame: np.ndarray, path: str) -> None:
        """Write a JPEG screenshot to disk."""
        try:
            cv2.imwrite(path, frame)
            log.info("Screenshot saved: %s", path)
        except Exception:
            log.exception("Failed to save screenshot: %s", path)

    def _audio(self, message: str, threat_level: str) -> None:
        """Play alarm/siren and speak the message via TTS."""
        # ── Sound effect ─────────────────────────────────────────────────
        if self._mixer_ok:
            try:
                import pygame
                sound_file = config.SIREN_SOUND if threat_level == "CRITICAL" else config.ALARM_SOUND
                if Path(sound_file).is_file():
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    # Wait for playback (max 5 seconds)
                    time.sleep(min(sound.get_length(), 5.0))
                else:
                    log.warning("Sound file not found: %s", sound_file)
            except Exception:
                log.exception("Audio playback error.")

        # ── TTS voice ────────────────────────────────────────────────────
        with self._tts_lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 120)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
            except Exception:
                log.exception("TTS engine error.")


# ══════════════════════════════════════════════════════════════════════════════
# M9 — Display Functions
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (255, 200, 0),   # amber
    "weapon": (0, 0, 255),     # red
    "bag":    (0, 165, 255),   # orange
}

THREAT_COLORS: dict[str, tuple[int, int, int]] = {
    "CRITICAL":   (0, 0, 255),
    "HIGH":       (0, 140, 255),
    "SUSPICIOUS": (0, 220, 255),
    "SAFE":       (0, 200, 0),
}


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> None:
    """Draw coloured bounding boxes for raw detections."""
    for d in detections:
        color = CATEGORY_COLORS.get(d["category"], (200, 200, 200))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f'{d["label"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracked_persons: list[dict[str, Any]],
    tracker: ObjectTracker,
    behaviour: BehaviourFlags,
    threat_color: tuple[int, int, int],
) -> None:
    """Draw tracked person boxes with behaviour tags and trajectories."""
    for p in tracked_persons:
        tid = p["track_id"]
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), threat_color, 2)

        # Build label with behaviour tags
        tags: list[str] = [f"P#{tid}"]
        if tid in behaviour.loitering:
            lvl = behaviour.loitering_level.get(tid, "")
            tags.append(f"[LOITER-{lvl}]")
        if tid in behaviour.fast_move:
            tags.append("[FAST]")
        if tid in behaviour.geofence_breach:
            tags.append("[BREACH]")
        # (weapon proximity tags — removed)

        label = " ".join(tags)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), threat_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Trajectory polyline
        if config.SHOW_TRAJECTORIES and tid in tracker.person_history:
            pts = list(tracker.person_history[tid])
            if len(pts) >= 2:
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    c = (
                        int(threat_color[0] * alpha),
                        int(threat_color[1] * alpha),
                        int(threat_color[2] * alpha),
                    )
                    cv2.line(frame, pts[k - 1], pts[k], c, 1, cv2.LINE_AA)


def draw_bag_tracks(
    frame: np.ndarray,
    tracked_bags: list[dict[str, Any]],
    tracker: ObjectTracker,
    behaviour: BehaviourFlags,
    scene_label: str,
    behav_analyser: BehaviourAnalyser | None = None,
    tracked_persons: list[dict[str, Any]] | None = None,
) -> None:
    """Draw tracked bag boxes with abandoned-bag indicators."""
    # Build lookup for abandoned bags (only flag them visually in high-risk scenes)
    high_risk_scenes = ("railway", "subway", "airport", "station", "terminal", "hospital","dorm_room","staircase")
    is_high_risk = any(kw in scene_label.lower() for kw in high_risk_scenes)

    abandoned_lookup: dict[int, AbandonedBag] = {}
    if is_high_risk:
        abandoned_lookup = {ab.bag_id: ab for ab in behaviour.abandoned_bags}

    # Build person lookup for owner line
    person_lookup: dict[int, dict[str, Any]] = {}
    if tracked_persons:
        for p in tracked_persons:
            person_lookup[p["track_id"]] = p

    frame_n = int(time.time() * 10)  # for flashing effect

    for b in tracked_bags:
        bid = b["track_id"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        if bid in abandoned_lookup:
            ab = abandoned_lookup[bid]
            # Flashing red outline
            if frame_n % 2 == 0:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 3)
            else:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 180), 2)

            ab_label = f"ABANDONED {ab.duration:.0f}s [{ab.level}]"
            cv2.putText(frame, ab_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        label = f"BAG#{bid}"
        cv2.putText(frame, label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

        # Draw line from bag to associated owner if in Monitoring state
        if behav_analyser and bid in behav_analyser.bag_states:
            bs = behav_analyser.bag_states[bid]
            if bs.state == "Monitoring" and bs.owner_id is not None:
                owner = person_lookup.get(bs.owner_id)
                if owner is not None:
                    bag_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    owner_center = (owner["cx"], owner["cy"])
                    cv2.line(frame, bag_center, owner_center,
                             (0, 200, 0), 1, cv2.LINE_AA)


def draw_geofence_zones(frame: np.ndarray) -> None:
    """Draw semi-transparent red polygon overlays for each geofence zone."""
    for zone in config.GEOFENCE_ZONES:
        polygon = np.array(zone["polygon"], dtype=np.int32)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (0, 0, 200))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Border
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

        # Zone name at centroid
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon[0][0], polygon[0][1]

        name = zone.get("name", "RESTRICTED")
        cv2.putText(frame, name.upper(), (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_behaviour_flags(frame: np.ndarray, behaviour: BehaviourFlags, scene_label: str, frame_height: int) -> None:
    """Draw bottom-left stacked behaviour flag indicators."""
    flags: list[tuple[str, tuple[int, int, int]]] = []

    if behaviour.geofence_breach:
        flags.append(("ZONE BREACH", (0, 0, 255)))
    if behaviour.abandoned_bags:
        high_risk_scenes = ("railway", "subway", "airport", "station", "terminal", "hospital","dorm_room","staircase")
        if any(kw in scene_label.lower() for kw in high_risk_scenes):
            max_lvl = max((ab.level for ab in behaviour.abandoned_bags),
                          key=lambda l: {"CRITICAL": 3, "HIGH": 2, "SUSPICIOUS": 1}.get(l, 0),
                          default="SUSPICIOUS")
            flags.append((f"ABANDONED BAG [{max_lvl}]", THREAT_COLORS.get(max_lvl, (0, 220, 255))))

    if behaviour.confirmed_weapons:
        flags.append((f"WEAPON CONFIRMED", (0, 0, 255)))
    if behaviour.fast_move:
        flags.append(("FAST MOVE", (0, 140, 255)))
    if behaviour.loitering:
        max_lvl = max(behaviour.loitering_level.values(),
                      key=lambda l: {"CRITICAL": 3, "HIGH": 2, "SUSPICIOUS": 1}.get(l, 0),
                      default="SUSPICIOUS")
        flags.append((f"LOITERING [{max_lvl}]", THREAT_COLORS.get(max_lvl, (0, 220, 255))))
    if behaviour.crowd:
        flags.append(("CROWD", (0, 140, 255)))

    y = frame_height - 12
    for text, color in flags:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (4, y - th - 4), (tw + 12, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y -= th + 12


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_n: int,
    scene_results: list[tuple[str, float]],
    threat_result: ThreatResult,
) -> None:
    """Draw top-left semi-transparent HUD with system status."""
    # Semi-transparent overlay
    hud_h = 110
    hud_w = 420
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 18
    line_h = 18

    # Line 1: FPS and frame
    cv2.putText(frame, f"FPS:{fps:.1f}  Frame:{frame_n}", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_h

    # Line 2: Scene
    if scene_results:
        slabel, sconf = scene_results[0]
        scene_text = f"Scene: {slabel} ({sconf:.2f})"
    else:
        scene_text = "Scene: unknown"
    cv2.putText(frame, scene_text, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    y += line_h

    # Line 3: Threat level
    t_color = THREAT_COLORS.get(threat_result.level, (200, 200, 200))
    cv2.putText(frame, f"THREAT: {threat_result.level} ({threat_result.score:.2f})",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, t_color, 2, cv2.LINE_AA)
    y += line_h + 4

    # Line 4: Matched rule (small)
    rule_text = threat_result.matched_rule[:70] if threat_result.matched_rule else ""
    cv2.putText(frame, rule_text, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    y += line_h - 2

    # Line 5: First reason (small)
    if threat_result.reasons:
        reason = threat_result.reasons[0][:65]
        cv2.putText(frame, reason, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Initialise all modules and run the surveillance main loop."""
    log.info("═" * 60)
    log.info("Real-Time Intelligent Surveillance System — Starting")
    log.info("═" * 60)

    # ── Initialise modules ───────────────────────────────────────────────
    video = VideoSource(config.VIDEO_SOURCE, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    detector = ObjectDetector()
    tracker = ObjectTracker()
    behav = BehaviourAnalyser()
    scene = SceneClassifier()
    threat = ThreatAnalyser()
    alerts = AlertManager()

    # Wait for the first frame to arrive
    time.sleep(0.3)

    frame_n = 0
    fps_deque: deque[float] = deque(maxlen=30)
    prev_time = time.time()

    log.info("Entering main loop. Press 'Q' to quit.")

    try:
        while video.is_alive():
            frame = video.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_n += 1

            # ── FPS calculation ──────────────────────────────────────────
            now = time.time()
            elapsed = now - prev_time
            prev_time = now
            if elapsed > 0:
                fps_deque.append(1.0 / elapsed)
            fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0.0

            # ── Detection ────────────────────────────────────────────────
            detections = detector.detect(frame, frame_n)

            # ── Tracking ─────────────────────────────────────────────────
            tracked_persons, tracked_bags = tracker.update(detections)

            # ── Behaviour analysis ───────────────────────────────────────
            behav_flags = behav.analyse(
                tracked_persons, tracked_bags, detections, fps, frame,
            )

            # ── Scene classification ─────────────────────────────────────
            if frame_n % config.SCENE_INFER_EVERY == 0:
                scene.submit(frame)
            scene_results = scene.result
            scene_label = scene_results[0][0] if scene_results else "unknown"
            is_restricted = scene.is_restricted()

            # ── Threat assessment ────────────────────────────────────────
            threat_result = threat.analyse(
                detections, behav_flags, scene_label, is_restricted,
                tracked_persons, tracked_bags,
            )

            # ── Alert dispatch ───────────────────────────────────────────
            alerts.dispatch(threat_result, behav_flags, is_restricted, scene_label, frame)

            # ── Drawing ──────────────────────────────────────────────────
            draw_geofence_zones(frame)
            draw_detections(frame, detections)
            draw_bag_tracks(frame, tracked_bags, tracker, behav_flags, scene_label,
                            behav, tracked_persons)
            draw_tracks(frame, tracked_persons, tracker, behav_flags,
                        THREAT_COLORS.get(threat_result.level, (200, 200, 200)))
            draw_behaviour_flags(frame, behav_flags, scene_label, config.FRAME_HEIGHT)
            draw_hud(frame, fps, frame_n, scene_results, threat_result)

            # ── Display ──────────────────────────────────────────────────
            cv2.imshow("Surveillance System  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit requested by user.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
    finally:
        video.release()
        cv2.destroyAllWindows()
        log.info("Surveillance system stopped.")


if __name__ == "__main__":
    main()
