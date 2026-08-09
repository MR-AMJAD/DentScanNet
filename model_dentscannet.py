# model_dentscannet.py — DentScanNet architecture (Amjadian et al.)
# Blocks: MKIRBlock (Eq.2), MKIRABlock (Eq.9), GCAMBlock (Eq.3-5),
#         LFGate / LFG-s k=11, LFG-d k=7 (Eq.6-7), CReFGate (Eq.8)
# Import CUSTOM_OBJECTS when loading saved checkpoints.

import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_HEIGHT = 256
IMAGE_WIDTH  = 256

# Default pixel calibration for clinical measurements (iGH, iGR, iABL).
# This value is system- and crop-specific and should be measured for your setup.
# See README for calibration instructions.
PIXELS_PER_MM = 37

ALL_FEATURES = ['GM', 'CEJ', 'ABC', 'TOOTH', 'BONE', 'GINGIVA']
POINT_FEATURES = ['GM', 'CEJ', 'ABC']
REGION_FEATURES = ['TOOTH', 'BONE', 'GINGIVA']
NUM_CLASSES     = 2
CH_BASE         = [16, 32, 64, 96]

class ResizeToMatch(layers.Layer):
    """Bilinear-resize inputs[0] to the spatial dims of inputs[1]."""
    def call(self, inputs):
        x, ref = inputs
        s = tf.shape(ref)
        return tf.image.resize(x, [s[1], s[2]], method='bilinear')

    def get_config(self):
        return super().get_config()

def conv_bn_relu(x, ch, kernel=3, stride=1, tag=''):
    """Conv → BN → ReLU building primitive."""
    x = layers.Conv2D(ch, kernel, strides=stride, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      name=f'{tag}_c')(x)
    x = layers.BatchNormalization(name=f'{tag}_bn')(x)
    return layers.ReLU(name=f'{tag}_r')(x)

class MKIRBlock(layers.Layer):
    """Encoder block: expand -> parallel DW(3,5,7) -> fuse -> project + residual (Eq. 2)."""

    def __init__(self, filters, expansion=2, **kw):
        super().__init__(**kw)
        self.filters   = filters
        self.expansion = expansion

    def build(self, input_shape):
        f, mid = self.filters, self.filters * self.expansion
        in_ch  = input_shape[-1]
        self.exp    = layers.Conv2D(mid, 1, use_bias=False,
                                    kernel_initializer='he_normal')
        self.exp_bn = layers.BatchNormalization()
        self.dw3 = layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.dw5 = layers.DepthwiseConv2D(5, padding='same', use_bias=False)
        self.dw7 = layers.DepthwiseConv2D(7, padding='same', use_bias=False)
        self.mix    = layers.Conv2D(mid, 1, use_bias=False,
                                    kernel_initializer='he_normal')
        self.mix_bn = layers.BatchNormalization()
        self.proj    = layers.Conv2D(f, 1, use_bias=False,
                                     kernel_initializer='he_normal')
        self.proj_bn = layers.BatchNormalization()
        if in_ch != f:
            self.sc    = layers.Conv2D(f, 1, use_bias=False,
                                       kernel_initializer='he_normal')
            self.sc_bn = layers.BatchNormalization()
        else:
            self.sc = self.sc_bn = None
        super().build(input_shape)

    def call(self, x, training=None):
        sc = self.sc_bn(self.sc(x), training=training) if self.sc else x
        h  = tf.nn.relu(self.exp_bn(self.exp(x), training=training))
        mk = self.dw3(h) + self.dw5(h) + self.dw7(h)
        mk = tf.nn.relu(self.mix_bn(self.mix(mk), training=training))
        return tf.nn.relu(self.proj_bn(self.proj(mk), training=training) + sc)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'expansion': self.expansion})
        return cfg

class MKIRABlock(layers.Layer):
    """Decoder fusion: MKIR -> channel SE -> spatial attention (Eq. 9)."""

    def __init__(self, filters, expansion=2, reduction=4, **kw):
        super().__init__(**kw)
        self.filters   = filters
        self.expansion = expansion
        self.reduction = reduction

    def build(self, input_shape):
        f, inner = self.filters, max(self.filters // self.reduction, 8)
        in_ch    = input_shape[-1]
        if in_ch != f:
            self.adapt    = layers.Conv2D(f, 1, use_bias=False,
                                          kernel_initializer='he_normal')
            self.adapt_bn = layers.BatchNormalization()
        else:
            self.adapt = self.adapt_bn = None
        self.mkir    = MKIRBlock(f, expansion=self.expansion)
        self.se1     = layers.Dense(inner, activation='relu',
                                    kernel_initializer='he_normal')
        self.se2     = layers.Dense(f, activation='sigmoid',
                                    kernel_initializer='glorot_normal')
        self.sp_dw   = layers.DepthwiseConv2D(7, padding='same', use_bias=False)
        self.sp_gate = layers.Conv2D(1, 1, activation='sigmoid',
                                     use_bias=False,
                                     kernel_initializer='glorot_normal')
        super().build(input_shape)

    def call(self, x, training=None):
        if self.adapt:
            x = tf.nn.relu(self.adapt_bn(self.adapt(x), training=training))
        h  = self.mkir(x, training=training)
        sq = tf.reduce_mean(h, axis=[1, 2])
        ch = tf.reshape(self.se2(self.se1(sq)), [-1, 1, 1, self.filters])
        h  = h * ch
        sp = self.sp_gate(self.sp_dw(h))
        return h * sp

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'expansion': self.expansion,
                    'reduction': self.reduction})
        return cfg

class GCAMBlock(layers.Layer):
    """
    Global Context Aggregation Module (Section IV.D, Eq. 3–5).

    Bottleneck block that aggregates directional, edge-magnitude, and
    multi-scale spatial context without global pooling, preserving spatial
    resolution for the decoder.  Adapted from SCSegamba SAVSSBlock
    (Liu et al., 2025, ref [33]).
    """

    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.dim = dim

    def build(self, input_shape):
        d = self.dim
        # Step 1 — directional context (Eq. 3a–3e)
        self.dwH  = layers.DepthwiseConv2D((1, 7), padding='same', use_bias=False)
        self.dwV  = layers.DepthwiseConv2D((7, 1), padding='same', use_bias=False)
        self.dwD1 = layers.DepthwiseConv2D(3, dilation_rate=2,
                                            padding='same', use_bias=False)
        self.dwD2 = layers.DepthwiseConv2D(3, dilation_rate=4,
                                            padding='same', use_bias=False)
        self.dir_fuse = layers.Conv2D(d, 1, padding='same', use_bias=False,
                                       kernel_initializer='he_normal')
        self.dir_bn   = layers.BatchNormalization()
        # Step 2 — edge-magnitude spatial gate (Eq. 4a–4c)
        self.eH     = layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.eV     = layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.e_bn   = layers.BatchNormalization()
        self.e_gate = layers.Conv2D(1, 1, padding='same', activation='sigmoid',
                                    kernel_initializer='glorot_normal')
        # Step 3 — multi-scale refinement with selective gate (Eq. 5a–5e)
        self.s1    = layers.DepthwiseConv2D(3, dilation_rate=1,
                                             padding='same', use_bias=False)
        self.s2    = layers.DepthwiseConv2D(3, dilation_rate=2,
                                             padding='same', use_bias=False)
        self.s3    = layers.DepthwiseConv2D(3, dilation_rate=4,
                                             padding='same', use_bias=False)
        self.s_mix  = layers.Conv2D(d, 1, padding='same', use_bias=False,
                                    kernel_initializer='he_normal')
        self.s_gate = layers.Conv2D(d, 1, padding='same',
                                    kernel_initializer='zeros',
                                    bias_initializer='ones')
        self.s_bn   = layers.BatchNormalization()
        self.norm1  = layers.LayerNormalization()
        self.norm2  = layers.LayerNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        fh  = tf.nn.relu(self.dwH(x));  fv  = tf.nn.relu(self.dwV(x))
        fd1 = tf.nn.relu(self.dwD1(x)); fd2 = tf.nn.relu(self.dwD2(x))
        df  = tf.nn.relu(self.dir_bn(
              self.dir_fuse(fh + fv + fd1 + fd2), training=training))
        eh  = self.eH(x); ev = self.eV(x)
        mag = tf.sqrt(tf.square(eh) + tf.square(ev) + 1e-6)
        mag = self.e_bn(mag, training=training)
        eg  = self.e_gate(mag)
        st  = df * eg
        xn  = self.norm1(st)
        so  = self.s_bn(
              self.s_mix(self.s1(xn) + self.s2(xn) + self.s3(xn)),
              training=training)
        sg  = tf.nn.sigmoid(self.s_gate(xn))
        so  = sg * tf.nn.relu(so) + (1.0 - sg) * xn
        return self.norm2(x + so)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'dim': self.dim})
        return cfg

class LFGate(layers.Layer):
    """
    Local Feature Gate (Section IV.E, Eq. 6–7).

    Large-kernel depthwise spatial gate followed by channel
    squeeze-and-excitation.  Used with two different kernel sizes:
      LFG-s  k=11  gates encoder skip connections before concatenation
      LFG-d  k=7   first step of LFCR in the decoder (Section IV.H)
    """

    def __init__(self, filters, spatial_k=11, reduction=4, **kw):
        super().__init__(**kw)
        self.filters   = filters
        self.spatial_k = spatial_k
        self.reduction = reduction

    def build(self, input_shape):
        f, inner = self.filters, max(self.filters // self.reduction, 8)
        in_ch    = input_shape[-1]
        if in_ch != f:
            self.adapt    = layers.Conv2D(f, 1, use_bias=False,
                                          kernel_initializer='he_normal')
            self.adapt_bn = layers.BatchNormalization()
        else:
            self.adapt = self.adapt_bn = None
        self.lk_dw   = layers.DepthwiseConv2D(self.spatial_k, padding='same',
                                               use_bias=False)
        self.lk_bn   = layers.BatchNormalization()
        self.lk_gate = layers.Conv2D(f, 1, padding='same',
                                     activation='sigmoid', use_bias=False,
                                     kernel_initializer='glorot_normal')
        self.se1 = layers.Dense(inner, activation='relu',
                                kernel_initializer='he_normal')
        self.se2 = layers.Dense(f, activation='sigmoid',
                                kernel_initializer='glorot_normal')
        self.bn  = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        if self.adapt:
            x = tf.nn.relu(self.adapt_bn(self.adapt(x), training=training))
        lg   = tf.nn.relu(self.lk_bn(self.lk_dw(x), training=training))
        gate = self.lk_gate(lg)
        x    = x * gate
        sq   = tf.reduce_mean(x, axis=[1, 2])
        ch   = tf.reshape(self.se2(self.se1(sq)), [-1, 1, 1, self.filters])
        return self.bn(x * ch, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'spatial_k': self.spatial_k,
                    'reduction': self.reduction})
        return cfg

class CReFGate(layers.Layer):
    """Channel-only gate: GAP -> FC(C/4) -> FC(C) -> sigmoid (Eq. 8). No spatial component."""

    def __init__(self, filters, reduction=4, **kw):
        super().__init__(**kw)
        self.filters   = filters
        self.reduction = reduction

    def build(self, input_shape):
        inner = max(self.filters // self.reduction, 8)
        self.fc1 = layers.Dense(inner, activation='relu',
                                kernel_initializer='he_normal')
        self.fc2 = layers.Dense(self.filters, activation='sigmoid',
                                kernel_initializer='glorot_normal')
        super().build(input_shape)

    def call(self, x, training=None):
        gap  = tf.reduce_mean(x, axis=[1, 2])
        gate = self.fc2(self.fc1(gap))
        return x * tf.reshape(gate, [-1, 1, 1, self.filters])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'reduction': self.reduction})
        return cfg

def build_encoder(inp, ch):

    s0 = conv_bn_relu(inp, ch[0], stride=2, tag='stem')

    s1 = MKIRBlock(ch[0], name='e1a')(s0)
    s1 = MKIRBlock(ch[0], name='e1b')(s1)

    s1d = conv_bn_relu(s1, ch[1], stride=2, tag='s1_dn')
    s2  = MKIRBlock(ch[1], name='e2a')(s1d)
    s2  = MKIRBlock(ch[1], name='e2b')(s2)

    s2d = conv_bn_relu(s2, ch[2], stride=2, tag='s2_dn')
    s3  = MKIRBlock(ch[2], name='e3a')(s2d)
    s3  = MKIRBlock(ch[2], name='e3b')(s3)

    s3d = conv_bn_relu(s3, ch[3], stride=2, tag='s3_dn')
    s4  = MKIRBlock(ch[3], name='e4a')(s3d)
    s4  = MKIRBlock(ch[3], name='e4b')(s4)

    return s1, s2, s3, s4

def build_decoder(s4, s3, s2, s1, ch, input_shape, features, num_classes):
    # LFCR decoder: LFG-s(skip) -> concat -> MKIRA -> LFG-d -> CReF at each stage (Eq. 10)
    s4 = GCAMBlock(ch[3], name='gcam')(s4)

    def gate_skip(skip, c, stage):
        return LFGate(c, spatial_k=11, name=f'{stage}_lfg_s')(skip)

    def apply_lfcr(feat, c, stage):
        feat = LFGate(c, spatial_k=7, name=f'{stage}_lfg_d')(feat)
        feat = CReFGate(c,             name=f'{stage}_cref')(feat)
        return feat

    sk3 = gate_skip(s3, ch[2], 'd3')
    x3  = layers.UpSampling2D(2, interpolation='bilinear', name='d3_up')(s4)
    x3  = ResizeToMatch(name='d3_rs')([x3, sk3])
    x3  = layers.Concatenate(name='d3_cat')([x3, sk3])
    d3  = MKIRABlock(ch[2], name='d3_mkira')(x3)
    d3  = apply_lfcr(d3, ch[2], 'd3')

    sk2 = gate_skip(s2, ch[1], 'd2')
    x2  = layers.UpSampling2D(2, interpolation='bilinear', name='d2_up')(d3)
    x2  = ResizeToMatch(name='d2_rs')([x2, sk2])
    x2  = layers.Concatenate(name='d2_cat')([x2, sk2])
    d2  = MKIRABlock(ch[1], name='d2_mkira')(x2)
    d2  = apply_lfcr(d2, ch[1], 'd2')

    sk1 = gate_skip(s1, ch[0], 'd1')
    x1  = layers.UpSampling2D(2, interpolation='bilinear', name='d1_up')(d2)
    x1  = ResizeToMatch(name='d1_rs')([x1, sk1])
    x1  = layers.Concatenate(name='d1_cat')([x1, sk1])
    d1  = MKIRABlock(ch[0], name='d1_mkira')(x1)
    d1  = apply_lfcr(d1, ch[0], 'd1')

    shared = layers.UpSampling2D(2, interpolation='bilinear', name='final_up')(d1)
    shared = layers.Resizing(input_shape[0], input_shape[1],
                               interpolation='bilinear',
                               name='final_rs')(shared)
    shared = layers.Conv2D(16, 3, padding='same', use_bias=False,
                           kernel_initializer='he_normal',
                           name='shared_conv')(shared)
    shared = layers.BatchNormalization(name='shared_bn')(shared)
    shared = layers.ReLU(name='shared_relu')(shared)

    outputs = {}
    for feat in features:
        h = layers.Conv2D(32, 3, padding='same',
                          name=f'{feat}_head_conv')(shared)
        h = layers.BatchNormalization(name=f'{feat}_head_bn')(h)
        h = layers.Activation('relu', name=f'{feat}_head_relu')(h)
        outputs[feat] = layers.Conv2D(
            num_classes, 1, activation='softmax',
            name=f'{feat}_output')(h)

    return outputs

def build_dentscannet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                       features=None,
                       num_classes=NUM_CLASSES,
                       ch=None):

    if features is None:
        features = ALL_FEATURES
    if ch is None:
        ch = CH_BASE

    inp = tf.keras.Input(input_shape, name='input')
    s1, s2, s3, s4 = build_encoder(inp, ch)
    outputs = build_decoder(s4, s3, s2, s1, ch,
                             input_shape, features, num_classes)
    return models.Model(inp, list(outputs.values()), name='DentScanNet')

CUSTOM_OBJECTS = {
    'MKIRBlock': MKIRBlock,
    'MKIRABlock': MKIRABlock,
    'GCAMBlock': GCAMBlock,
    'LFGate': LFGate,
    'CReFGate': CReFGate,
    'ResizeToMatch': ResizeToMatch,
}
# Pass to tf.keras.models.load_model(..., custom_objects=CUSTOM_OBJECTS)
