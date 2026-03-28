
# About the Dataset

The dataset used in this project is the Carparts-seg dataset (by Ultralytics). It provides instances of car images paired with polygon segmentation boundaries.

- Classes:Total of 23 distinct car parts including:
  `back_bumper`, `back_door`, `back_glass`, `back_left_door`, `back_left_light`, `back_light`, `back_right_door`, `back_right_light`, `front_bumper`, `front_door`, `front_glass`, `front_left_door`, `front_left_light`, `front_light`, `front_right_door`, `front_right_light`, `hood`, `left_mirror`, `object`, `right_mirror`, `tailgate`, `trunk`, `wheel`.
- Splits:
  - 3,516 Training Images
  - 276 Validation Images
  - 401 Test Images
- Format: The raw labels are initially presented as YOLO polygon data points.

# Process

Here is a summary of the steps and workflow executed during this project:

# 1. Dataset Checking and Validation
- Used `check_dataset.py` and `validate_labels.py` to count and cross-verify the presence of image files against their corresponding label and mask entries. This helped ensure there were no missing training labels.

# 2. Mask Generation (Preprocessing)
- The raw dataset supplies labels as text files representing polygon coordinates.
- Using `create_val_masks.py`, these polygonal coordinates were converted into strict 2D array mask image `.png` files suitable for semantic segmentation mapping. We used OpenCV (`cv2.fillPoly`) to handle this conversion dynamically.

# 3. Data Loading Pipeline
- Designed `dataset.py` to prepare custom batch data structures.
- During training, images are processed by rescaling resolution to uniformly sized inputs (`256x256`) and scaled down by dividing pixel values by `255.0` to normalize them for the neural network.

# 4. Model Architectural Design
- In `model.py`, a custom U-Net architecture was built completely from scratch using `tf.keras`.
- The architecture implements a classic symmetric approach spanning:
  - Encoder Blocks: Convolutional (`Conv2D`), `BatchNormalization`, and `MaxPool2D` layers to capture spatial context.
  - Bottleneck Block:Extracting deeper abstract features.
  - Decoder Blocks: Upsampling (`Conv2DTranspose`) and Concatenating with skip connections to rebuild higher resolution spatial output maps.
- The model outputs a multi-channel map matching the exact `num_classes` passing through a `softmax` activation to predict the most likely car part pixel-by-pixel.

# 5. Model Training
- The `train.py` script orchestrates compiling and fitting the actual model.
- Used the `adam` optimizer and `sparse_categorical_crossentropy` loss (as our pixel labels are effectively discrete integers scaling `num_classes`).
- Trained the model securely using both the prepared Train and Validation inputs. Following successful training runs, models were sequentially persisted to disk (like `unet_model_1epoch.h5`).

# 6. Testing, Evaluation, and Visualization
- Tested the effectiveness of the semantic segmentation algorithm locally using `test.py`.
- Evaluated performance using key performance indices such as Pixel Accuracy and Mean IoU (Intersection over Union) metrics on random validation subsets.
- Handled visual sanity checks with Matplotlib, showing an end-to-end grid comparison of the: 1) Original Input Image, 2) Ground True Mask and 3) Our Model's Predicted Mask.
