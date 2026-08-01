import cv2
import numpy as np
import os

# -------------------------------------------------
# Vermillion v0.2
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

    # -------------------------------------------------
    # Convert to Grayscale
    # -------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # Edge Detection (Canny)
    # -------------------------------------------------
    edges = cv2.Canny(gray, 100, 200)

    # -------------------------------------------------
    # Corner Detection (Shi-Tomasi)
    # -------------------------------------------------
    corner_img = img.copy()

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=10
    )

    corner_count = 0

    if corners is not None:
        corners = np.intp(corners)
        corner_count = len(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corner_img, (x, y), 4, (0, 0, 255), -1)

    # -------------------------------------------------
    # Calculate Edge Density
    # -------------------------------------------------
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = edge_pixels / total_pixels

    # -------------------------------------------------
    # Rule-Based Terrain Classification
    # -------------------------------------------------
    if edge_density < 0.05:
        status = "🟢 SAFE"
    elif edge_density < 0.12:
        status = "🟡 CAUTION"
    else:
        status = "🔴 DANGER"

    # -------------------------------------------------
    # Print Results
    # -------------------------------------------------
    print("\n----------------------------------")
    print("Image          :", filename)
    print("Edge Density   :", round(edge_density, 3))
    print("Corner Count   :", corner_count)
    print("Terrain Status :", status)

    # -------------------------------------------------
    # Display Images
    # -------------------------------------------------
    cv2.imshow("Original Image", img)
    cv2.imshow("Grayscale Image", gray)
    cv2.imshow("Detected Edges", edges)
    cv2.imshow("Detected Corners", corner_img)

    cv2.waitKey(0)

cv2.destroyAllWindows()

print("\nAnalysis Complete.")
