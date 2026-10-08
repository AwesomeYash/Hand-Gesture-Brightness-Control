# Hand Gesture Brightness Control (Windows)

Control your screen brightness with your hand in front of the webcam. The program tracks 21 hand landmarks with MediaPipe and turns simple gestures into brightness changes.

| Gesture | Action |
|---|---|
| Pinch thumb and index finger together | Lower brightness |
| Spread thumb and index finger apart | Raise brightness (distance 25 px to 200 px maps to 0 % to 100 %) |
| Fist | Snap brightness to 10 % |
| Open palm (fingers spread) | Snap brightness to 100 % |

Press `q` to quit.

## How it works

1. **OpenCV** captures video from the webcam.
2. **MediaPipe Hands** (lite model, one hand) finds the hand landmarks.
3. The pinch distance is the pixel distance between thumb tip (landmark 4) and index fingertip (landmark 8).
4. Fist and open palm are detected from the ratio of each fingertip's distance from the wrist to its knuckle's distance from the wrist (fist below 0.9, open palm above 1.7), so detection does not depend on hand size or distance from the camera.
5. The target value is smoothed with an exponential filter (`SMOOTH_ALPHA = 0.25`) and only sent to the system when it changes enough, using `screen_brightness_control`. The call runs in a background thread because Windows (WMI) brightness calls can block.

## Setup

1. Windows with Python 3.9 to 3.11 (MediaPipe's legacy `solutions` API is used).
2. Install dependencies (there is no `requirements.txt` yet):

```bash
pip install opencv-python mediapipe numpy screen-brightness-control
```

3. Run:

```bash
python "Brightness Control/brightness.py"
```

Only displays that Windows can control (laptop panels and monitors supporting DDC/CI) will change. If the webcam does not open, close other apps using it or change the camera index in the script.

## Tuning

Constants near the top of `brightness.py`: `MIN_DIST` / `MAX_DIST` (pinch range in pixels), `FIST_RATIO` / `PALM_RATIO`, `FIST_BRIGHTNESS` / `PALM_BRIGHTNESS`, and `SMOOTH_ALPHA` (lower is smoother but laggier).

## Repository contents

```
Brightness Control/brightness.py   the application
Progress.txt                       development notes
gesture_ui_state_machine.png       design diagram for a multi-screen menu (planned, not implemented)
```

## Planned: multi-screen menu

`gesture_ui_state_machine.png` sketches a two-state interface: a fist toggles a menu in which the index fingertip is a cursor and a pinch selects **Laptop screen**, **Display screen** or **Both screens**; brightness control is paused while the menu is visible. This design is **not implemented** in the current code.

## Limitations

- Windows only; no tests, demo video or measured latency yet.
- One hand, one set of thresholds: lighting and hand angle can cause false fist/palm detections.
- `Progress.txt` is shared with the sibling volume-control project and its first entries describe volume control.

## Related projects

[Hand-Gesture-Volume-Control](https://github.com/AwesomeYash/Hand-Gesture-Volume-Control) and [Hand-Gesture-Filters-Control](https://github.com/AwesomeYash/Hand-Gesture-Filters-Control) use the same hand-tracking approach.

## License

MIT License (see `LICENSE`).
