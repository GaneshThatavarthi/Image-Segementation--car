import os
img_path = "images/train"
label_path = "labels/train"
print("Images:", len(os.listdir(img_path)))
print("Labels:", len(os.listdir(label_path)))
images = set([f.split('.')[0] for f in os.listdir(img_path)])
labels = set([f.split('.')[0] for f in os.listdir(label_path)])
missing = images - labels
print("Missing labels:", missing)