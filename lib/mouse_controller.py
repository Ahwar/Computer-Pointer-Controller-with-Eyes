"""
Mouse controller utilities built on top of pyautogui.

Provides a lightweight MouseController class to drive the system pointer using
relative movements that are scaled by a precision factor and animated over a
configurable duration. This is useful for driving the pointer from gaze- or
pose-estimation outputs where you receive small x, y offsets.

Constructor presets (string keys accepted):
- precision: "high" (100), "medium" (500), "low" (1000)
- speed: "fast" (1), "medium" (5), "slow" (10)

API summary:
- MouseController.move(x, y)
    Move the pointer by (x * precision, -y * precision). Note the y value is
    negated internally to accommodate screen coordinate orientation.
- MouseController.move_to_center()
    Move the pointer to the center of the primary screen using the same scaling.

Notes and requirements:
- This module disables pyautogui.FAILSAFE by default. Re-enable it if you want
  the corner-escape behavior.
- Install pyautogui before use.
- Interpret x and y according to your gaze model (e.g., normalized offsets,
  pixel deltas, etc.) and tune precision/speed presets accordingly.

Example:
    mc = MouseController(precision="medium", speed="fast")
    mc.move(0.1, -0.05)  # relative movement scaled by precision
"""

import pyautogui

pyautogui.FAILSAFE = False


class MouseController:
    def __init__(self, precision, speed):
        precision_dict = {"high": 100, "low": 1000, "medium": 500}
        speed_dict = {"fast": 1, "slow": 10, "medium": 5}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]

    def move(self, x, y):
        pyautogui.moveRel(
            x * self.precision, -1 * y * self.precision, duration=self.speed
        )

    def move_to_center(self):
        w, h = pyautogui.size()
        cx, cy = w // 2, h // 2
        cur_x, cur_y = pyautogui.position()

        # required inputs for mc.move
        dx = (cx - cur_x) / self.precision
        dy = (
            cur_y - cy
        ) / self.precision  # invert because MouseController.move negates y internally

        self.move(dx, dy)
