"""
Hand Border - control the color of a border around your webcam feed with your hand.

  OPEN hand + TWIST (like turning a dial)  ->  border cycles through the rainbow
  Close into a FIST                        ->  color locks
  OPEN again                               ->  unlocks, picks up where it left off
  PEACE sign (index + middle up, rest down) ->  turns purple and sticks until you clearly twist again

Keys:  q or Esc = quit    r = reset to red

The border mirrors the color, and your Govee lights follow it too (see govee_lights.py).
"""

import colorsys
import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from govee_lights import GoveeLights

# ----------------------------------------------------------------- settings
BORDER = 40            # border thickness in pixels
TWIST_GAIN = 2.0       # hue degrees per degree of twist (2 = a 180 deg twist covers the whole rainbow)
SMOOTHING = 0.5        # 0 = twitchy, 0.9 = very smooth but laggy
CONFIRM_FRAMES = 3     # frames a fist/open must be seen before it counts (stops flicker)
PURPLE_HUE = 280.0     # hue the peace sign jumps to
STICKY_TWIST = 20.0    # degrees you must twist before purple lets go

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# MediaPipe gives 21 points per hand. The ones we care about:
#   0 = wrist, 9 = base of middle finger
#   finger tips 8, 12, 16, 20 and the middle joints below them 6, 10, 14, 18
WRIST, MIDDLE_BASE = 0, 9
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]   # (tip, middle joint) for index..pinky
PEACE = [True, True, False, False]                  # index + middle up, ring + pinky down
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
              (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ----------------------------------------------------------------- reading the hand

def open_fingers(hand):
    """[index, middle, ring, pinky] - a finger is 'open' if its tip is farther
    from the wrist than its middle joint."""
    wrist = hand[WRIST]
    dist = lambda p: math.hypot(p.x - wrist.x, p.y - wrist.y)
    return [dist(hand[tip]) > dist(hand[joint]) * 1.1 for tip, joint in FINGERS]


def twist_angle(hand):
    """Angle of the hand, wrist -> middle finger. 0 = fingers straight up, + = tilted right."""
    w, m = hand[WRIST], hand[MIDDLE_BASE]
    return math.degrees(math.atan2(m.x - w.x, w.y - m.y))


def angle_diff(a, b):
    """Smallest signed difference a - b, handling the wrap at +/-180."""
    return (a - b + 180) % 360 - 180


class ColorDial:
    """Turns hand twist into a hue, and fists into a lock."""

    def __init__(self):
        self.hue = 0.0            # 0..360, 0 = red
        self.locked = False
        self.angle = None         # smoothed hand angle
        self.last_angle = None    # angle at the last hue update
        self.candidate, self.candidate_frames = None, 0
        self.sticky = False       # True after a peace sign: hold purple until a real twist

    def update(self, hand):
        opened = open_fingers(hand)
        fingers = sum(opened)

        if opened == PEACE:
            seen = "peace"
        elif fingers <= 1:
            seen = "fist"
        elif fingers >= 3:
            seen = "open"
        else:
            seen = None  # half-closed: don't change anything

        # Only switch state after CONFIRM_FRAMES frames in a row agree
        needs_change = ((seen == "fist" and not self.locked)
                        or (seen == "open" and self.locked)
                        or (seen == "peace" and not (self.locked and self.hue == PURPLE_HUE)))
        if needs_change:
            if self.candidate == seen:
                self.candidate_frames += 1
            else:
                self.candidate, self.candidate_frames = seen, 1
            if self.candidate_frames >= CONFIRM_FRAMES:
                self.locked = seen != "open"
                if seen == "peace":
                    self.hue = PURPLE_HUE
                    self.sticky = True
                self.last_angle = None   # re-anchor so unlocking doesn't jump
                self.candidate = None
        else:
            self.candidate = None

        # Smooth the raw angle a bit to remove jitter
        raw = twist_angle(hand)
        if self.angle is None or self.last_angle is None:
            self.angle = raw
        else:
            self.angle += (1 - SMOOTHING) * angle_diff(raw, self.angle)

        # Twisting while open nudges the hue by how much the hand rotated since last time.
        if not self.locked:
            if self.last_angle is None:
                self.last_angle = self.angle
            else:
                delta = angle_diff(self.angle, self.last_angle)
                if self.sticky:
                    # Hold purple through hand-opening and jitter; a deliberate twist releases it
                    if abs(delta) > STICKY_TWIST:
                        self.sticky = False
                        self.last_angle = self.angle   # start turning from purple, no jump
                elif abs(delta) > 0.4:  # ignore tiny wobbles
                    self.hue = (self.hue + delta * TWIST_GAIN) % 360
                    self.last_angle = self.angle
        return fingers

    def hand_lost(self):
        self.last_angle = None
        self.candidate = None

    def bgr(self):
        r, g, b = colorsys.hsv_to_rgb(self.hue / 360, 1, 1)
        return int(b * 255), int(g * 255), int(r * 255)   # OpenCV uses BGR order


# ----------------------------------------------------------------- drawing

def outlined_text(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_hand(frame, hand, color):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand]
    for a, b in HAND_LINES:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)
    # the line we measure the twist from
    cv2.line(frame, pts[WRIST], pts[MIDDLE_BASE], color, 4, cv2.LINE_AA)


def draw_rainbow_bar(img, hue, y):
    """Thin rainbow along the bottom with a marker showing the current hue."""
    w = img.shape[1]
    x0, x1 = 20, w - 20
    for x in range(x0, x1):
        r, g, b = colorsys.hsv_to_rgb((x - x0) / (x1 - x0), 1, 1)
        cv2.line(img, (x, y), (x, y + 8), (int(b * 255), int(g * 255), int(r * 255)))
    mx = int(x0 + hue / 360 * (x1 - x0))
    cv2.rectangle(img, (mx - 3, y - 4), (mx + 3, y + 12), (255, 255, 255), 2)


# ----------------------------------------------------------------- main loop

def main():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand tracking model (~7 MB, first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    ))

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cam.isOpened():
        raise SystemExit("Couldn't open the webcam.")

    dial = ColorDial()
    lights = GoveeLights(start_color=(255, 0, 0))   # connects in the background, starts red
    start = time.monotonic()
    cv2.namedWindow("Hand Border", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, so moving right moves right on screen

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = tracker.detect_for_video(image, int((time.monotonic() - start) * 1000))

        fingers = 0
        hand = result.hand_landmarks[0] if result.hand_landmarks else None
        if hand:
            fingers = dial.update(hand)
        else:
            dial.hand_lost()

        if hand:
            draw_hand(frame, hand, dial.bgr())

        # Status text
        if not hand:
            status = "Show your hand"
        elif dial.sticky:
            status = "PURPLE  (twist to change)"
        elif dial.locked:
            status = "LOCKED  (open hand to unlock)"
        else:
            status = "LIVE  (twist to change, fist to lock)"
        outlined_text(frame, status, (15, 30))
        outlined_text(frame, f"fingers open: {fingers}   hue: {dial.hue:.0f}", (15, 60), 0.6)
        outlined_text(frame, lights.status(), (15, 90), 0.6)
        draw_rainbow_bar(frame, dial.hue, frame.shape[0] - 25)

        lights.set_color(dial.bgr()[::-1])   # BGR -> RGB

        # The border = our pretend LED strip
        framed = cv2.copyMakeBorder(frame, BORDER, BORDER, BORDER, BORDER,
                                    cv2.BORDER_CONSTANT, value=dial.bgr())
        if dial.locked:
            # small lock badge in the top-right of the border
            outlined_text(framed, "LOCKED", (framed.shape[1] - 120, 28), 0.7)

        cv2.imshow("Hand Border", framed)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            dial.hue = 0.0
            dial.sticky = False

    lights.close()
    cam.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
