import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random

img_dir = "images/train"
label_dir = "labels/train"
img_files = os.listdir(img_dir)
random_img_file = random.choice(img_files)

img_path = os.path.join(img_dir, random_img_file)
label_path = os.path.join(label_dir, random_img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

img = cv2.imread(img_path)
if img is not None:
    h, w, _ = img.shape
    
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                data = list(map(float, line.split()))
                points = data[1:]
                
                pts = []
                for i in range(0, len(points), 2):
                    x = int(points[i] * w)
                    y = int(points[i+1] * h)
                    pts.append([x, y])
                
                pts = np.array(pts, np.int32)
                cv2.polylines(img, [pts], True, (0,255,0), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(random_img_file)
    plt.axis("off") 
    plt.savefig("visualized_sample.png")
    print("Visualized sample saved to 'visualized_sample.png'")
else:
    print("Failed to load image:", img_path)