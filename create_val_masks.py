import os
import cv2
import numpy as np

# Change paths to target the Validation folder instead of Training!
img_dir = "images/val"
label_dir = "labels/val"
mask_dir = "masks/val"

# Create mask folder for validation
os.makedirs(mask_dir, exist_ok=True)

for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    label_file = img_file.replace(".jpg", ".txt").replace(".png", ".txt")
    label_path = os.path.join(label_dir, label_file)

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                data = list(map(float, line.split()))
                cls = int(data[0])   
                points = data[1:]

                pts = []
                for i in range(0, len(points), 2):
                    x = int(points[i] * w)
                    y = int(points[i+1] * h)
                    pts.append([x, y])

                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], cls + 1)

    save_path = os.path.join(mask_dir, img_file.replace(".jpg", ".png"))
    cv2.imwrite(save_path, mask)

print("Validation Masks created successfully!")
