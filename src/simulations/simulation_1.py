import cv2
import numpy as np
import random

class Simulation:
    def __init__(self):
        # Parameters
        self.width, self.height = 800, 600
        self.num_balloons = 10
        self.balloon_radius_range = (15, 50)
        self.balloon_speed = 5

        # Create a list to hold balloon properties
        self.balloons = []

        # Initialize random positions and colors
        for _ in range(self.num_balloons):
            radius = random.randint(*self.balloon_radius_range)
            color = (0, 0, 255) if random.random() < 0.5 else (255, 0, 0)  # Red or Blue
            x = random.randint(radius, self.width - radius)
            y = random.randint(radius, self.height - radius)
            dx = random.choice([-self.balloon_speed, self.balloon_speed])  # Random horizontal speed
            dy = random.choice([-self.balloon_speed, self.balloon_speed])  # Random vertical speed
            self.balloons.append((x, y, radius, color, dx, dy))

    
    def run(self):
        # Create a blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Update balloon positions and draw them
        for i, (x, y, radius, color, dx, dy) in enumerate(self.balloons):
            # Update position
            x += dx
            y += dy

            # Bounce off walls
            if x - radius < 0 or x + radius > self.width:
                dx = -dx
            if y - radius < 0 or y + radius > self.height:
                dy = -dy
            
            # Update the balloon properties
            self.balloons[i] = (x, y, radius, color, dx, dy)

            # Draw the balloon
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
        return img

'''sim = Simulation()
    
while True:
    img = sim.run()
    # Show the image
    cv2.imshow('Balloon Simulation', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()'''
