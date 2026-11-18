"""
════════════════════════════════════════════════════════════════════════════════
draw_geofence.py — Visual Geofence Zone Drawing Tool
════════════════════════════════════════════════════════════════════════════════

Run this tool to visually draw restricted zones on your camera feed.
Click to place vertices, then press keys to manage zones.

Usage:
    python draw_geofence.py

Controls:
    LEFT CLICK   — Place a vertex point
    RIGHT CLICK  — Undo last vertex
    ENTER        — Finish current zone (must have >= 3 points)
    N            — Name the zone (type in terminal, press Enter)
    S            — Save all zones to config.py and quit
    R            — Reset current zone (clear all points)
    D            — Delete last saved zone
    Q / ESC      — Quit without saving
"""

import os
import re
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import config

# ── State ─────────────────────────────────────────────────────────────────────
current_points: list[tuple[int, int]] = []
saved_zones: list[dict] = []
zone_counter = 1
current_name = ""

# Pre-load existing zones from config
for z in config.GEOFENCE_ZONES:
    saved_zones.append(z)
    zone_counter += 1


def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    """Handle mouse clicks for vertex placement."""
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"  Vertex added: ({x}, {y})  [{len(current_points)} points]")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            removed = current_points.pop()
            print(f"  Vertex removed: {removed}  [{len(current_points)} points]")


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw all saved zones and current in-progress zone on the frame."""
    display = frame.copy()

    # Draw saved zones (green, semi-transparent)
    for zone in saved_zones:
        polygon = np.array(zone["polygon"], dtype=np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [polygon], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        cv2.polylines(display, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # Label at centroid
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = polygon[0][0], polygon[0][1]
        name = zone.get("name", "zone")
        cv2.putText(display, name.upper(), (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, name.upper(), (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1, cv2.LINE_AA)

    # Draw current zone being drawn (red)
    if current_points:
        for i, pt in enumerate(current_points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if len(current_points) >= 2:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(display, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        if len(current_points) >= 3:
            # Show preview of closed polygon
            pts = np.array(current_points, dtype=np.int32)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 200))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 0, 255), thickness=1,
                          lineType=cv2.LINE_AA)

    # ── HUD instructions ─────────────────────────────────────────────────
    hud_lines = [
        f"GEOFENCE DRAWING TOOL   |   Saved zones: {len(saved_zones)}",
        f"Current zone: {len(current_points)} vertices   (need >= 3)",
        "",
        "CLICK=add vertex  |  RIGHT-CLICK=undo  |  ENTER=finish zone",
        "N=name zone  |  S=SAVE & quit  |  R=reset  |  D=delete last  |  Q=quit",
    ]
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    for i, line in enumerate(hud_lines):
        cv2.putText(display, line, (8, 14 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    return display


def save_to_config() -> None:
    """Write the saved zones into config.py's GEOFENCE_ZONES list."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build the new GEOFENCE_ZONES value
    if not saved_zones:
        zones_str = "GEOFENCE_ZONES: list = []"
    else:
        lines = ["GEOFENCE_ZONES: list = ["]
        for zone in saved_zones:
            name = zone["name"]
            poly = zone["polygon"]
            poly_str = ", ".join(f"({x},{y})" for x, y in poly)
            lines.append(f'    {{"name": "{name}", "polygon": [{poly_str}]}},')
        lines.append("]")
        zones_str = "\n".join(lines)

    # Replace existing GEOFENCE_ZONES block — find start line and matching ']'
    lines_list = content.splitlines(keepends=True)
    start_idx = None
    end_idx = None
    bracket_depth = 0
    for i, line in enumerate(lines_list):
        if start_idx is None:
            if line.strip().startswith("GEOFENCE_ZONES"):
                start_idx = i
                bracket_depth += line.count("[") - line.count("]")
                if bracket_depth <= 0:
                    end_idx = i
                    break
        else:
            bracket_depth += line.count("[") - line.count("]")
            if bracket_depth <= 0:
                end_idx = i
                break

    if start_idx is not None and end_idx is not None:
        new_lines = lines_list[:start_idx] + [zones_str + "\n"] + lines_list[end_idx + 1:]
        new_content = "".join(new_lines)
    else:
        # Fallback: append to end
        new_content = content + "\n" + zones_str + "\n"

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n  Saved {len(saved_zones)} zone(s) to config.py!")
    for z in saved_zones:
        print(f"    - {z['name']}: {len(z['polygon'])} vertices")


def main() -> None:
    """Run the interactive geofence drawing tool."""
    global current_points, zone_counter, current_name

    print("=" * 60)
    print("  GEOFENCE ZONE DRAWING TOOL")
    print("=" * 60)
    print(f"  Video source: {config.VIDEO_SOURCE}")
    print(f"  Frame size: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"  Existing zones: {len(saved_zones)}")
    print()
    print("  Click on the video to place polygon vertices.")
    print("  Press ENTER when you have >= 3 points to save a zone.")
    print("  Press S to save all zones to config.py and quit.")
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Cannot open video source:", config.VIDEO_SOURCE)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    window_name = "Draw Geofence Zones  [S=Save  Q=Quit]"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display = draw_overlay(frame)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # Q or ESC
            print("\n  Quit without saving.")
            break

        elif key == 13:  # ENTER — finish current zone
            if len(current_points) < 3:
                print("  Need at least 3 vertices! Keep clicking.")
                continue
            name = current_name if current_name else f"zone_{zone_counter}"
            zone = {"name": name, "polygon": list(current_points)}
            saved_zones.append(zone)
            print(f"\n  Zone saved: '{name}' with {len(current_points)} vertices")
            print(f"  Polygon: {current_points}")
            zone_counter += 1
            current_points = []
            current_name = ""

        elif key == ord("n"):  # N — name the zone
            print("  Enter zone name: ", end="", flush=True)
            current_name = input().strip()
            if current_name:
                print(f"  Zone will be named: '{current_name}'")
            else:
                print("  Using default name.")

        elif key == ord("r"):  # R — reset current
            current_points = []
            current_name = ""
            print("  Current zone reset.")

        elif key == ord("d"):  # D — delete last saved zone
            if saved_zones:
                removed = saved_zones.pop()
                print(f"  Deleted zone: '{removed['name']}'")
            else:
                print("  No zones to delete.")

        elif key == ord("s"):  # S — save and quit
            save_to_config()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
