import os
import cv2
import numpy as np
from model import build_unet

# Paths
img_dir = "images/train"
mask_dir = "masks/train"

val_img_dir = "images/val"
val_mask_dir = "masks/val"

# Load images and masks
def load_data(img_dir, mask_dir, size=256):
    images = []
    masks = []
    
    # Optional: Limiting to first 100 images purely for speed if testing
    img_files = os.listdir(img_dir)
    
    for img_file in img_files:
        # Load image
        img = cv2.imread(os.path.join(img_dir, img_file))
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        img = img / 255.0

        # Load mask
        mask_file = img_file.replace(".jpg", ".png")
        mask = cv2.imread(os.path.join(mask_dir, mask_file), 0)
        if mask is None:
            continue
        mask = cv2.resize(mask, (size, size))

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

print("Loading Training dataset...")
X_train, Y_train = load_data(img_dir, mask_dir)

print("Loading Validation dataset...")
X_val, Y_val = load_data(val_img_dir, val_mask_dir)

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

# Build model
model = build_unet(input_shape=(256, 256, 3))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\nModel compiled successfully!")

# Set to exactly 1 epoch with batch size back to 4
epochs = 1
batch_size = 4

print(f"\nStarting training for {epochs} epoch with the dedicated validation set...")
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

# Save model
model.save("unet_model_1epoch.h5")
print("\nTraining complete and model saved as 'unet_model_1epoch.h5'!") 
      