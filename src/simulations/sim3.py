import cv2
import numpy as np
import random

class Stage3:
    def __init__(self):
        self.w, self.h = 800, 600
        self.lanes = [150, 400, 650]
        self.targets = []
        self.last_clear_time = None

    def spawn(self):
        if len(self.targets) >= 3:
            if self.last_clear_time is None:
                self.last_clear_time = cv2.getTickCount()
            return

        if self.last_clear_time is not None:
            elapsed = (cv2.getTickCount() - self.last_clear_time) / cv2.getTickFrequency()
            if elapsed < 5.0:
                return
            self.last_clear_time = None

        occupied = {int(t[0]) for t in self.targets}
        available = [l for l in self.lanes if l not in occupied]
        if not available:
            return
        lane = random.choice(available)
        color = (0, 0, 255) if random.random() < 0.5 else (255, 0, 0)
        shape = random.choice(['circle', 'square', 'triangle'])
        self.targets.append([lane, self.h // 2, 10, 0, color, shape])

    def draw_target(self, img, x, y, size, color, shape):
        x, y, size = int(x), int(y), int(size)
        if shape == 'circle':
            cv2.circle(img, (x, y), size, color, -1)
        elif shape == 'square':
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
        elif shape == 'triangle':
            pts = np.array([
                [x, y - size],
                [x - size, y + size],
                [x + size, y + size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)

    def out_of_bounds(self, x, y, size, shape):
        size = int(size)
        return x - size < 0 or x + size > self.w or y - size < 0 or y + size > self.h

    def run(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if random.random() < 0.05:
            self.spawn()

        new_targets = []

        for x, y, size, dx, color, shape in self.targets:
            if dx == 0:
                size += 0.5

            if size > 70:
                dx = -10 if x < self.w // 2 else 10

            x += dx

            if self.out_of_bounds(x, y, size, shape):
                continue

            new_targets.append([x, y, size, dx, color, shape])
            self.draw_target(img, x, y, size, color, shape)

        self.targets = new_targets

        # turret
        cv2.circle(img, (self.w // 2, self.h - 30), 10, (0, 255, 0), -1)

        return img


if __name__ == "__main__":
    sim = Stage3()

    while True:
        img = sim.run()
        cv2.imshow("Stage3", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()