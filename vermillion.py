import cv2
import numpy as np
import os

# -------------------------------------------------
# Vermillion v0.1
# AI-Guided Terrain Traversability Analyzer
# -------------------------------------------------

# Always locate the images folder relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "images")

# Check if the folder exists
if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Images folder not found!\nExpected location:\n{IMAGE_FOLDER}")
    exit()

# Get all JPG, JPEG and PNG images
image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(image_files) == 0:
    print("❌ No images found inside the images folder.")
    exit()

print("=" * 50)
print("VERMILLION TERRAIN ANALYSIS")
print("=" * 50)

for filename in image_files:

    path = os.path.join(IMAGE_FOLDER, filename)

    img = cv2.imread(path)

    if img is None:
        print(f"⚠ Could not read {filename}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # Calculate Edge Density
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = edge_pixels / total_pixels

    # Safety Classification
    if edge_density < 0.05:
        status = "🟢 SAFE"
    elif edge_density < 0.12:
        status = "🟡 CAUTION"
    else:
        status = "🔴 DANGER"

    print("\n----------------------------------")
    print("Image:", filename)
    print("Edge Density:", round(edge_density, 3))
    print("Terrain Status:", status)

    # Display image
    cv2.imshow("Original", img)
    cv2.imshow("Detected Edges", edges)

    cv2.waitKey(0)

cv2.destroyAllWindows()

print("\nAnalysis Complete.")
