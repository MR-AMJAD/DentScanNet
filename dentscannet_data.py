# dentscannet_data.py
#
# DentScanNet — Data Loading and Preprocessing
# Provides the generator-based data pipeline used to train DentScanNet
# (Amjadian et al.).
#
# Expected directory layout:
#   data_dir/
#     train/
#       images/   ← .png / .jpg frames (256×256 or resized on load)
#       masks/    ← binary masks named  {frame_stem}_{FEATURE}.png
#     val/
#       images/
#       masks/
#     test/
#       images/
#       masks/
#
# Mask naming convention:
#   For a frame  "img_0042.png"  the GM mask is  "img_0042_GM.png".
#   Missing mask files are treated as all-background for that feature.
#
# Point landmark masks (GM, CEJ, ABC) are optionally dilated with a
# circular structuring element (radius = dilation_radius pixels, default 4)
# before one-hot encoding, matching the protocol in Section V.A of the paper.

import os
import math

import cv2
import numpy as np
import tensorflow as tf

from model_dentscannet import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    ALL_FEATURES, POINT_FEATURES, REGION_FEATURES, NUM_CLASSES,
)

def preprocess_mask(mask, feature,
                    enable_point_dilation=True,
                    dilation_radius=4):
    """
    Convert a raw grayscale mask to a normalised binary float32 map.

    Point features are optionally dilated with a circular kernel (radius=dilation_radius).
    """
    if isinstance(mask, tf.Tensor):
        mask = mask.numpy()

    mask = np.array(mask, dtype=np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0

    binary = (mask > 0.5).astype(np.float32)

    if feature in POINT_FEATURES and enable_point_dilation:
        if np.any(binary):
            r = int(dilation_radius)
            y, x = np.ogrid[-r:r + 1, -r:r + 1]
            kernel = (x * x + y * y <= r * r).astype(np.uint8)
            binary = cv2.dilate(
                binary.astype(np.uint8), kernel, iterations=1
            ).astype(np.float32)

    return binary.astype(np.float32)

def create_data_generator(batch_size, target_size,
                           train_path, val_path,
                           features=None,
                           enable_point_dilation=True,
                           dilation_radius=4,
                           seed=42):
    """
    Create generator-based train and validation data pipelines.

    A seeded numpy Generator instance is used so that augmentation sequences
    are fully reproducible regardless of call order.

    Augmentation (training only):
      - Horizontal flip  p = 0.5

"""
    if features is None:
        features = ALL_FEATURES

    rng = np.random.default_rng(seed)

    def _generator(data_path, is_training):
        images_path = os.path.join(data_path, 'images')
        masks_path  = os.path.join(data_path, 'masks')
        image_files = sorted([
            f for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        while True:
            if is_training:
                rng.shuffle(image_files)

            for i in range(0, len(image_files), batch_size):
                batch_files  = image_files[i:i + batch_size]
                batch_images = []
                batch_masks  = {feat: [] for feat in features}

                for fname in batch_files:
                    try:
                        # Load image and normalise to [0, 1]
                        img = tf.keras.preprocessing.image.load_img(
                            os.path.join(images_path, fname),
                            target_size=target_size)
                        img = (tf.keras.preprocessing.image
                               .img_to_array(img).astype(np.float32) / 255.0)

                        stem   = os.path.splitext(fname)[0]
                        fmasks = {}

                        for feat in features:
                            mask_path = os.path.join(
                                masks_path, f'{stem}_{feat}.png')

                            if os.path.exists(mask_path):
                                m = tf.keras.preprocessing.image.load_img(
                                    mask_path,
                                    target_size=target_size,
                                    color_mode='grayscale')
                                m = (tf.keras.preprocessing.image
                                     .img_to_array(m).squeeze())
                                proc = preprocess_mask(
                                    m, feat,
                                    enable_point_dilation=enable_point_dilation,
                                    dilation_radius=dilation_radius)
                            else:
                                # Missing mask → all-background
                                proc = np.zeros(
                                    target_size, dtype=np.float32)

                            oh = np.zeros(
                                (*target_size, NUM_CLASSES), dtype=np.float32)
                            oh[..., 1] = proc
                            oh[..., 0] = 1.0 - proc
                            fmasks[feat] = oh

                        # Augmentation: horizontal flip (training only)
                        if is_training and rng.random() > 0.5:
                            img = img[:, ::-1]
                            for feat in features:
                                fmasks[feat] = fmasks[feat][:, ::-1]

                        # Sanity: clip and replace any NaN / Inf
                        img = np.nan_to_num(
                            np.clip(img, 0.0, 1.0), nan=0.0).astype(np.float32)

                        batch_images.append(img)
                        for feat in features:
                            batch_masks[feat].append(fmasks[feat])

                    except Exception as e:
                        print(f'  Skipping {fname}: {e}')
                        continue

                if batch_images:
                    yield (
                        np.array(batch_images, dtype=np.float32),
                        {f'{feat}_output': np.array(batch_masks[feat],
                                                     dtype=np.float32)
                         for feat in features}
                    )

    # Count files to compute steps
    def _count(path):
        img_dir = os.path.join(path, 'images')
        return len([f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    n_train = _count(train_path)
    n_val   = _count(val_path)
    print(f'  [DataGen] Train: {n_train} | Val: {n_val}')

    return (
        _generator(train_path, is_training=True),
        _generator(val_path,   is_training=False),
        max(1, math.ceil(n_train / batch_size)),
        max(1, math.ceil(n_val   / batch_size)),
    )

def wrap_as_tf_dataset(generator, target_size, features=None):
    """
    Wrap a Python generator as a prefetched tf.data.Dataset.

    Parameters
    ----------
    generator   : Python generator yielding (images, masks_dict)
    target_size : tuple (H, W)
    features    : list; default ALL_FEATURES

    Returns
    -------
    tf.data.Dataset
    """
    if features is None:
        features = ALL_FEATURES

    sig = (
        tf.TensorSpec(shape=(None, target_size[0], target_size[1], 3),
                      dtype=tf.float32),
        {f'{f}_output': tf.TensorSpec(
             shape=(None, target_size[0], target_size[1], NUM_CLASSES),
             dtype=tf.float32)
         for f in features}
    )

    def _wrap():
        for batch in generator:
            yield batch

    return tf.data.Dataset.from_generator(
        _wrap, output_signature=sig).prefetch(2)
