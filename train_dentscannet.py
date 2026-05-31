# train_dentscannet.py
#
# DentScanNet — Training Script
# Trains the DentScanNet baseline (Amjadian et al.).
#
# Loss (Eq. 12–14):
#   Point features  (GM, CEJ, ABC)         : 0.5·FocalTversky + 0.5·FocalBCE(OHEM)  w=3.0
#   Region features (TOOTH, BONE, GINGIVA) : 0.5·Dice + 0.5·FocalTversky            w=1.0
#
# Optimizer  : Adam  lr=1e-4  β1=0.9  β2=0.999  ε=1e-7  clipnorm=1.0
# Augment    : horizontal flip p=0.5  |  point dilation radius=4 px
# Callbacks  : EarlyStopping(val_mean_dice, patience=15)
#              ReduceLROnPlateau(val_mean_dice, patience=10, factor=0.5)
# Determinism: SEED=42, TF_DETERMINISTIC_OPS=1

import os, math, argparse, random, time, json
from datetime import datetime

os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED']         = '42'

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print(f'Seed={SEED}  |  Deterministic ops ENABLED  |  float32 only')
except (AttributeError, tf.errors.UnimplementedError):
    print(f'Seed={SEED}  |  TF_DETERMINISTIC_OPS set (env)  |  float32 only')

mixed_precision.set_global_policy('float32')

from model_dentscannet import (
    build_dentscannet, CUSTOM_OBJECTS,
    IMAGE_HEIGHT, IMAGE_WIDTH,
    ALL_FEATURES, POINT_FEATURES, REGION_FEATURES, NUM_CLASSES, CH_BASE,
)
from dentscannet_data import create_data_generator, wrap_as_tf_dataset

def dice_loss(y_true, y_pred, smooth=1e-6):
    yt = tf.cast(tf.reshape(y_true[..., 1], [-1]), tf.float32)
    yp = tf.cast(tf.reshape(y_pred[..., 1], [-1]), tf.float32)
    inter = tf.reduce_sum(yt * yp)
    denom = tf.reduce_sum(yt) + tf.reduce_sum(yp)
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)

def focal_tversky_loss(y_true, y_pred,
                        alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    yt = tf.cast(tf.reshape(y_true[..., 1], [-1]), tf.float32)
    yp = tf.cast(tf.reshape(y_pred[..., 1], [-1]), tf.float32)
    TP = tf.reduce_sum(yt * yp)
    FP = tf.reduce_sum((1.0 - yt) * yp)
    FN = tf.reduce_sum(yt * (1.0 - yp))
    tv = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return tf.pow(1.0 - tv, gamma)

def focal_bce_loss(y_true, y_pred,
                   gamma=2.0, alpha=0.25, ohem_ratio=0.75, smooth=1e-7):
    yt = tf.cast(tf.reshape(y_true[..., 1], [-1]), tf.float32)
    yp = tf.clip_by_value(
        tf.cast(tf.reshape(y_pred[..., 1], [-1]), tf.float32),
        smooth, 1.0 - smooth)
    bce_pos   = -alpha       * tf.pow(1.0 - yp, gamma) * tf.math.log(yp)
    bce_neg   = -(1.0-alpha) * tf.pow(yp,       gamma) * tf.math.log(1.0 - yp)
    per_pixel = yt * bce_pos + (1.0 - yt) * bce_neg
    n_total   = tf.shape(per_pixel)[0]
    n_keep    = tf.maximum(
        tf.cast(tf.cast(n_total, tf.float32) * ohem_ratio, tf.int32), 1)
    rank = tf.argsort(
        tf.argsort(per_pixel, direction='DESCENDING', stable=True), stable=True)
    mask = tf.stop_gradient(tf.cast(rank < n_keep, tf.float32))
    return tf.reduce_sum(mask * per_pixel) / tf.cast(n_keep, tf.float32)

def point_loss(y_true, y_pred):
    """Point landmark loss (Eq. 13): 0.5·FocalTversky + 0.5·FocalBCE(OHEM)."""
    return 0.5 * focal_tversky_loss(y_true, y_pred) + 0.5 * focal_bce_loss(y_true, y_pred)

def region_loss(y_true, y_pred):
    """Region segmentation loss (Eq. 12): 0.5·Dice + 0.5·FocalTversky."""
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * focal_tversky_loss(y_true, y_pred)

def dice_coefficient_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) == 3:
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
    if len(y_pred.shape) == 3:
        y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=2)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denom = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))

class MeanDiceCallback(tf.keras.callbacks.Callback):
    """Aggregates per-output Dice into val_mean_dice each epoch."""
    def __init__(self):
        super().__init__()
        self.best_mean_dice = 0.0
        self.best_epoch     = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        vk = [k for k in logs
              if 'dice_coefficient_metric' in k and k.startswith('val_')]
        if vk:
            md = float(np.mean([logs[k] for k in vk]))
            logs['val_mean_dice'] = md
            if md > self.best_mean_dice:
                self.best_mean_dice = md
                self.best_epoch     = epoch
            tk = [k for k in logs
                  if 'dice_coefficient_metric' in k and not k.startswith('val_')]
            if tk:
                logs['mean_dice'] = float(np.mean([logs[k] for k in tk]))

def train(data_dir, output_dir,
          epochs=100, batch_size=4, learning_rate=1e-4,
          enable_point_dilation=True, dilation_radius=4,
          point_loss_weight=3.0):
    os.makedirs(output_dir, exist_ok=True)
    shape  = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    target = (IMAGE_HEIGHT, IMAGE_WIDTH)

    SEP = '=' * 70
    print(f'\n{SEP}\nDentScanNet — Training')
    print(f'  Input   : {IMAGE_HEIGHT}x{IMAGE_WIDTH}')
    print(f'  Loss pt : 0.5*FocalTversky + 0.5*FocalBCE(OHEM)  w={point_loss_weight}')
    print(f'  Loss rg : 0.5*Dice + 0.5*FocalTversky  w=1.0')
    print(f'  Seed    : {SEED}  |  Data: {data_dir}\n{SEP}')

    model = build_dentscannet(shape, ALL_FEATURES, NUM_CLASSES, CH_BASE)
    total_params = model.count_params()
    print(f'  Parameters: {total_params:,}  ({total_params/1e6:.3f} M)')

    train_path = os.path.join(data_dir, 'train')
    val_path   = os.path.join(data_dir, 'val')
    (train_gen, val_gen, train_steps, val_steps) = create_data_generator(
        batch_size, target, train_path, val_path,
        ALL_FEATURES, enable_point_dilation, dilation_radius, seed=SEED)
    train_ds = wrap_as_tf_dataset(train_gen, target)
    val_ds   = wrap_as_tf_dataset(val_gen,   target)

    losses, loss_weights, metrics_dict = {}, {}, {}
    for feat in ALL_FEATURES:
        out = f'{feat}_output'
        losses[out]       = point_loss if feat in POINT_FEATURES else region_loss
        loss_weights[out] = point_loss_weight if feat in POINT_FEATURES else 1.0
        metrics_dict[out] = [dice_coefficient_metric, 'accuracy']

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=losses,
                  loss_weights=loss_weights, metrics=metrics_dict)

    mean_dice_cb = MeanDiceCallback()
    callbacks = [
        mean_dice_cb,
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'DentScanNet_best.h5'),
            monitor='val_mean_dice', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mean_dice', mode='max', patience=15,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mean_dice', mode='max',
            factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'DentScanNet_training.csv')),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds, steps_per_epoch=train_steps,
        validation_data=val_ds, validation_steps=val_steps,
        epochs=epochs, callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0

    val_res      = model.evaluate(val_ds, steps=val_steps, verbose=0)
    metric_names = model.metrics_names
    model.save(os.path.join(output_dir, 'DentScanNet_final.h5'))

    feat_dice = {}
    for i, nm in enumerate(metric_names):
        if i >= len(val_res): continue
        for feat in ALL_FEATURES:
            if f'{feat}_output_dice_coefficient_metric' in nm:
                feat_dice[feat] = float(val_res[i])

    mean_pt  = float(np.mean([feat_dice.get(f, 0.0) for f in POINT_FEATURES]))
    mean_rg  = float(np.mean([feat_dice.get(f, 0.0) for f in REGION_FEATURES]))
    mean_all = float(np.mean(list(feat_dice.values()))) if feat_dice else 0.0

    summary = {
        'parameters_M': round(total_params/1e6, 3),
        'best_val_mean_dice': float(mean_dice_cb.best_mean_dice),
        'best_epoch': int(mean_dice_cb.best_epoch + 1),
        'mean_dice_point': mean_pt,
        'mean_dice_region': mean_rg,
        'feature_dice': feat_dice,
        'epochs_trained': len(history.history['loss']),
        'training_time_minutes': round(elapsed/60, 1),
    }
    sp = os.path.join(output_dir, 'DentScanNet_summary.json')
    with open(sp, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f'\n{SEP}\nDentScanNet — Complete')
    print(f'  Best Dice  : {mean_dice_cb.best_mean_dice:.4f}  '
          f'(epoch {mean_dice_cb.best_epoch+1})')
    print(f'  Pt-DSC     : {mean_pt:.4f}   Rg-DSC: {mean_rg:.4f}')
    print(f'  Time       : {elapsed/60:.1f} min')
    print(f'{SEP}')
    return summary

def main():
    parser = argparse.ArgumentParser(
        description='DentScanNet training (Amjadian et al.)')
    parser.add_argument('--data_dir',             type=str, required=True)
    parser.add_argument('--output_dir',           type=str, default='./dentscannet_out')
    parser.add_argument('--epochs',               type=int,   default=100)
    parser.add_argument('--batch_size',           type=int,   default=4)
    parser.add_argument('--learning_rate',        type=float, default=1e-4)
    parser.add_argument('--point_loss_weight',    type=float, default=3.0)
    parser.add_argument('--enable_point_dilation',
                        type=lambda x: x.lower() in ('true','1','yes'),
                        default=True, metavar='BOOL')
    parser.add_argument('--dilation_radius',      type=int,   default=4)
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        print(f'GPUs: {len(gpus)}')
    else:
        print('No GPU — using CPU')

    train(args.data_dir, args.output_dir, args.epochs, args.batch_size,
          args.learning_rate, args.enable_point_dilation, args.dilation_radius,
          args.point_loss_weight)

if __name__ == '__main__':
    main()
