import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

print("Loading trained model...")
# Load the trained model (Ensure the training has fully finished)
try:
    model = tf.keras.models.load_model("unet_model_1epoch.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model! Either model training is not complete yet, or the file doesn't exist. Error: {e}")
    exit()

# Setup paths (Testing on some images from the validation set)
val_img_dir = "images/val"
val_mask_dir = "masks/val"
size = 256
num_classes = 25 # Enough to cover all custom classes

# Pick exactly 5 random images to test
img_files = os.listdir(val_img_dir)
test_files = random.sample(img_files, min(5, len(img_files))) # randomly select 5

# Initialize metrics
total_accuracy = []
mean_iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

print(f"Starting specific evaluation on 5 completely random images out of {len(img_files)}...")

for idx, img_file in enumerate(test_files):
    # 1. Load and prepare original Image
    img_path = os.path.join(val_img_dir, img_file)
    img = cv2.imread(img_path)
    if img is None: 
        continue
    
    # Store original for viewing (OpenCV uses BGR, Matplotlib uses RGB)
    orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare image for model strictly maintaining what we did in `train.py`
    img_resized = cv2.resize(img, (size, size))
    input_img = img_resized / 255.0
    input_img = np.expand_dims(input_img, axis=0) # Add batch dimension -> (1, 256, 256, 3)

    # 2. Load True Mask
    mask_file = img_file.replace(".jpg", ".png")
    mask_path = os.path.join(val_mask_dir, mask_file)
    true_mask = cv2.imread(mask_path, 0) # Read in grayscale

    # 3. Predict Mask
    pred = model.predict(input_img, verbose=0)[0]
    
    # Since we used sparse_categorical_crossentropy, we take the class with the highest probability
    pred_mask = np.argmax(pred, axis=-1)

    # Calculate Evaluation Metrics
    if true_mask is not None:
        # Resize ground truth using INTER_NEAREST to preserve integers
        true_mask_resized = cv2.resize(true_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        
        # Pixel Accuracy
        correct_pixels = np.sum(true_mask_resized == pred_mask)
        total_pixels = size * size
        accuracy = correct_pixels / total_pixels
        total_accuracy.append(accuracy)
        
        # Mean IoU
        mean_iou_metric.reset_state()
        mean_iou_metric.update_state(true_mask_resized, pred_mask)
        iou = mean_iou_metric.result().numpy()
        print(f"[{idx+1}/5] {img_file} -> Accuracy: {accuracy*100:.2f}% | Mean IoU: {iou:.4f}")
    
    # 4. Plot Comparison Display (always show since we only test 5)
    plt.figure(figsize=(15, 5))

    # Plot Original Image
    plt.subplot(1, 3, 1)
    plt.title(f"Original Image")
    plt.imshow(orig_img)
    plt.axis("off")

    # Plot True Mask
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    if true_mask is not None:
        plt.imshow(true_mask_resized, cmap='jet', vmin=0, vmax=num_classes)
    else:
        plt.text(0.5, 0.5, "True Mask Missing")
    plt.axis("off")

    # Plot Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=num_classes)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

print("\nTesting script finished!")
if total_accuracy:
    print(f"Overall Average Pixel Accuracy for those 5 random images: {np.mean(total_accuracy)*100:.2f}%")
