import cv2
import numpy as np
import random

class Stage2:
    def __init__(self):
        self.w, self.h = 800, 600
        self.lanes = [150, 400, 650]
        self.targets = []

    def spawn(self):
        occupied = {int(t[0]) for t in self.targets}
        available = [l for l in self.lanes if l not in occupied]
        if not available:
            return
        lane = random.choice(available)
        self.targets.append([lane, self.h // 2, 10, 0])



    def run(self):
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if random.random() < 0.05:
            self.spawn()

        new_targets = []

        for x, y, r, dx in self.targets:
            r += 0.5

            if r > 50:
                dx = -10 if x < self.w // 2 else 10
                r = r - 0.5

            x += dx

            if x - r < 0 or x + r > self.w or y - r < 0 or y + r > self.h:
                continue

            new_targets.append([x, y, r, dx])
            cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), -1)

        self.targets = new_targets

        # turret
        cv2.circle(img, (self.w // 2, self.h - 30), 10, (0, 255, 0), -1)

        return img


if __name__ == "__main__":
    sim = Stage2()

    while True:
        img = sim.run()
        cv2.imshow("Stage2", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()