import cv2
import numpy as np
import os

folder = "images"

for filename in os.listdir(folder):

    if filename.endswith(".jpg"):

        path = os.path.join(folder, filename)

        img = cv2.imread(path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size

        edge_density = edge_pixels / total_pixels

        print("\n------------------")
        print(filename)
        print("Edge Density:", round(edge_density, 3))

        if edge_density < 0.05:
            print("SAFE")
        elif edge_density < 0.12:
            print("CAUTION")
        else:
            print("DANGER")