"""
Hand Gesture Brightness Control
--------------------------------
Pinch your thumb and index finger together to lower the brightness.
Spread them apart to raise the brightness.
Make a fist to snap brightness to 10%.
Open your palm (fingers spread) to snap brightness to 100%.

Controls:
    q  -> quit the program
"""

import math
import threading

import cv2
import mediapipe as mp
import numpy as np

import screen_brightness_control as sbc


# ---------------------------------------------------------------------------
# MediaPipe hand detection setup
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # 0 = lite model, much faster, small accuracy tradeoff
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ---------------------------------------------------------------------------
# Distance range calibration
# These are pixel distances between thumb tip and index tip.
# Adjust MIN_DIST / MAX_DIST if it feels too sensitive or not sensitive enough.
# ---------------------------------------------------------------------------
MIN_DIST = 25    # fingers touching -> 0% brightness
MAX_DIST = 200   # fingers fully spread -> 100% brightness

# Fist / open-palm detection thresholds. These are ratios comparing each
# fingertip's distance from the wrist against its own knuckle (MCP) joint's
# distance from the wrist. Ratio-based (not raw pixel distance) so it works
# regardless of hand size or distance from the camera.
FIST_RATIO = 0.9   # tip closer to wrist than 0.9x the knuckle distance -> curled
PALM_RATIO = 1.7   # tip farther from wrist than 1.7x the knuckle distance -> extended 1.3->1.7

FIST_BRIGHTNESS = 10
PALM_BRIGHTNESS = 100

# Exponential smoothing factor. Lower = smoother but more lag,
# higher = snappier but more jittery. 0.2-0.35 is a good starting range.
SMOOTH_ALPHA = 0.25

# Only push a new value to the OS if it moved at least this many percentage
# points since the last call. Cuts down on redundant/expensive calls that
# were causing the lag you noticed with unthrottled updates.
UPDATE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Reusable gesture-to-metric helpers
# ---------------------------------------------------------------------------
def landmark_dist(a, b):
    """Normalized-coordinate distance between two landmarks (0.0-1.0 space)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def get_pinch_distance(landmarks, w, h):
    """Pixel distance between thumb tip (4) and index fingertip (8),
    plus their pixel coordinates for drawing."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
    x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

    distance = math.hypot(x2 - x1, y2 - y1)
    return distance, (x1, y1), (x2, y2)


def map_to_percent(distance, min_dist=MIN_DIST, max_dist=MAX_DIST):
    """Map a pixel distance to a 0-100 percentage, clamped to range."""
    percent = np.interp(distance, [min_dist, max_dist], [0, 100])
    return float(np.clip(percent, 0, 100))


def is_fist(landmarks, ratio=FIST_RATIO):
    """True if index, middle, ring, and pinky are all curled toward the wrist."""
    wrist = landmarks[0]
    tips = (8, 12, 16, 20)
    mcps = (5, 9, 13, 17)

    curled = 0
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_dist = landmark_dist(landmarks[tip_idx], wrist)
        mcp_dist = landmark_dist(landmarks[mcp_idx], wrist)
        if tip_dist < mcp_dist * ratio:
            curled += 1

    return curled == 4


_brightness_busy = threading.Event()


def set_brightness_async(value):
    """Push a brightness value to the OS on a background thread.

    screen_brightness_control talks to Windows via WMI, which can block for
    50-150ms per call. Doing that on the main loop stalls frame capture and
    MediaPipe processing, which is what causes visible lag. This fires the
    call on a daemon thread and skips new calls while one is still in
    flight, so we never queue up overlapping WMI requests.
    """
    if _brightness_busy.is_set():
        return

    def worker():
        _brightness_busy.set()
        try:
            sbc.set_brightness(int(value))
        except Exception:
            pass
        finally:
            _brightness_busy.clear()

    threading.Thread(target=worker, daemon=True).start()


def is_open_palm(landmarks, ratio=PALM_RATIO):
    """True if index, middle, ring, and pinky are all extended away from the wrist."""
    wrist = landmarks[0]
    tips = (8, 12, 16, 20)
    mcps = (5, 9, 13, 17)

    extended = 0
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_dist = landmark_dist(landmarks[tip_idx], wrist)
        mcp_dist = landmark_dist(landmarks[mcp_idx], wrist)
        if tip_dist > mcp_dist * ratio:
            extended += 1

    return extended == 4


def main():
    # CAP_DSHOW is generally faster/more responsive than the default backend
    # on Windows. Lower resolution than before (still plenty for hand
    # tracking) and a buffer size of 1 so we always grab the newest frame
    # instead of processing a backlog of stale ones.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that it's connected and not in use.")
        return

    # Seed the smoothed value with the display's current brightness so the
    # first frame doesn't jump.
    smoothed_percent = sbc.get_brightness()[0]
    last_set_percent = smoothed_percent

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w, _ = frame.shape

        # Color conversion: OpenCV uses BGR internally; MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        brightness_percent = None
        gesture_label = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark

                # --- Gesture priority: fist > open palm > pinch distance ---
                if is_fist(landmarks):
                    target_percent = FIST_BRIGHTNESS
                    gesture_label = "Fist: 10%"

                elif is_open_palm(landmarks):
                    target_percent = PALM_BRIGHTNESS
                    gesture_label = "Open palm: 100%"

                else:
                    distance, (x1, y1), (x2, y2) = get_pinch_distance(landmarks, w, h)

                    # Draw markers and connecting line for the pinch
                    cv2.circle(frame, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

                    if distance < MIN_DIST + 10:
                        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

                    target_percent = map_to_percent(distance)
                    gesture_label = "Pinch"

                # --- Exponential smoothing to kill jitter/lag ---
                smoothed_percent = (
                    SMOOTH_ALPHA * target_percent
                    + (1 - SMOOTH_ALPHA) * smoothed_percent
                )
                brightness_percent = smoothed_percent

                # --- Throttle: only call the OS if the change is meaningful ---
                if abs(smoothed_percent - last_set_percent) >= UPDATE_THRESHOLD:
                    set_brightness_async(smoothed_percent)
                    last_set_percent = smoothed_percent

        # --- Draw the brightness bar UI ---
        bar_x, bar_y, bar_w, bar_h = 50, 150, 35, 300
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 3)

        if brightness_percent is not None:
            filled_h = int(np.interp(brightness_percent, [0, 100], [bar_h, 0]))
            cv2.rectangle(
                frame,
                (bar_x, bar_y + filled_h),
                (bar_x + bar_w, bar_y + bar_h),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                frame, f'{int(brightness_percent)} %', (bar_x - 10, bar_y + bar_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            )
            if gesture_label:
                cv2.putText(
                    frame, gesture_label, (bar_x - 10, bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
        else:
            # Show current display brightness when no hand is detected
            current = int(last_set_percent)
            filled_h = int(np.interp(current, [0, 100], [bar_h, 0]))
            cv2.rectangle(
                frame,
                (bar_x, bar_y + filled_h),
                (bar_x + bar_w, bar_y + bar_h),
                (100, 100, 100),
                cv2.FILLED,
            )
            cv2.putText(
                frame, "No hand", (bar_x - 20, bar_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        cv2.putText(
            frame, "Press 'q' to quit", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.imshow("Hand Gesture Brightness Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()