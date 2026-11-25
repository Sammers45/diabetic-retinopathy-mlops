"""Network definitions (TensorFlow/Keras 3).

Every model takes RGB uint8-range images (0-255, as cached by ``dr.preprocess``) and exposes:

* a layer named ``embedding``, used by the drift monitor (PCA of embeddings), and
* a sigmoid output named ``p_referable`` (float32 even under mixed precision).

Design notes (from earlier mistakes):

* A pretrained backbone is always called with ``training=False`` so its BatchNorm layers stay in
  inference mode and keep their ImageNet statistics. With small data, letting BatchNorm update
  destroys the pretrained features. BatchNorm layers also stay frozen when the top of the backbone
  is unfrozen for fine-tuning.
* A *randomly initialised* backbone (``pretrained=False``, offline tests only) cannot learn while
  frozen, so it is trained end to end from the start, with BatchNorm in normal training mode.
* Augmentation lives inside the model, so it can never be applied at inference time by mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import keras
from keras import layers

BACKBONES = {
    "efficientnet_b0": keras.applications.EfficientNetB0,
    "efficientnet_b3": keras.applications.EfficientNetB3,
}
FAMILIES = (*BACKBONES, "baseline_cnn")
BACKBONE_NAME = "backbone"
EMBEDDING = "embedding"
OUTPUT = "p_referable"


def augmentation(cfg_aug: Mapping[str, Any], seed: int) -> keras.Sequential:
    """In-model augmentation: flips, full rotation, zoom, brightness, contrast.

    Fill value 128 matches the neutral grey that preprocessing puts outside the fundus, so a
    rotated image looks like a real preprocessed one.
    """
    aug: list[layers.Layer] = []
    if cfg_aug.get("flip", True):
        aug.append(layers.RandomFlip("horizontal_and_vertical", seed=seed))
    if cfg_aug.get("rotation", True):
        aug.append(layers.RandomRotation(1.0, fill_mode="constant", fill_value=128.0, seed=seed + 1))
    if cfg_aug.get("zoom"):
        aug.append(layers.RandomZoom(float(cfg_aug["zoom"]), fill_mode="constant", fill_value=128.0, seed=seed + 2))
    if cfg_aug.get("brightness"):
        aug.append(layers.RandomBrightness(float(cfg_aug["brightness"]), value_range=(0, 255), seed=seed + 3))
    if cfg_aug.get("contrast"):
        aug.append(layers.RandomContrast(float(cfg_aug["contrast"]), value_range=(0, 255), seed=seed + 4))
    return keras.Sequential(aug, name="augmentation")


def _baseline_backbone(size: int) -> keras.Model:
    """Small from-scratch CNN: the "is transfer learning worth it?" baseline."""
    inp = keras.Input((size, size, 3))
    x = layers.Rescaling(1.0 / 255.0)(inp)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    return keras.Model(inp, x, name=BACKBONE_NAME)


def build_backbone(family: str, size: int, pretrained: bool) -> keras.Model:
    if family == "baseline_cnn":
        return _baseline_backbone(size)
    if family not in BACKBONES:
        raise ValueError(f"model.family={family!r}; expected one of {FAMILIES}")
    # Keras derives the ImageNet weight file name from the model name, so the application must
    # keep its default name; it is then wrapped (sharing its layers) under our fixed name.
    net = BACKBONES[family](
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(size, size, 3),
        pooling="avg",
    )
    return keras.Model(net.input, net.output, name=BACKBONE_NAME)


def build_model(cfg: Mapping[str, Any], pretrained: bool | None = None) -> keras.Model:
    """Assemble augmentation, backbone and head from the config."""
    mcfg = cfg["model"]
    size = int(cfg["preprocess"]["image_size"])
    family = str(mcfg["family"])
    pretrained = bool(mcfg.get("pretrained", True)) if pretrained is None else pretrained
    pretrained = pretrained and family != "baseline_cnn"
    seed = int(cfg["project"]["seed"])

    backbone = build_backbone(family, size, pretrained)
    # Pretrained: keep BatchNorm in inference mode forever. Random init: normal BN behaviour.
    bn_inference = pretrained

    inp = keras.Input((size, size, 3), name="image")
    x = augmentation(cfg.get("augment", {}), seed)(inp)
    x = backbone(x, training=False) if bn_inference else backbone(x)
    x = layers.Dropout(float(mcfg.get("dropout", 0.3)), seed=seed)(x)
    x = layers.Dense(int(mcfg.get("embedding_dim", 256)), activation="relu", name=EMBEDDING)(x)
    x = layers.Dropout(float(mcfg.get("dropout", 0.3)), seed=seed + 1)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name=OUTPUT)(x)
    model = keras.Model(inp, out, name=f"dr_{family}")
    model.pretrained_backbone = pretrained  # read by the trainer to pick the training schedule
    return model


def get_backbone(model: keras.Model) -> keras.Model:
    return model.get_layer(BACKBONE_NAME)


def freeze_backbone(model: keras.Model) -> None:
    """Stage 1: only the head (embedding + output) trains."""
    get_backbone(model).trainable = False


def unfreeze_top(model: keras.Model, n_layers: int) -> int:
    """Stage 2: unfreeze the top ``n_layers`` backbone layers, keeping every BatchNorm frozen.

    Returns the number of trainable backbone layers, so it can be logged.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - int(n_layers))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    return sum(1 for layer in backbone.layers if layer.trainable)


def embedding_model(model: keras.Model) -> keras.Model:
    """Sub-model mapping an image to (embedding, p_referable).

    Caution: Keras 3 random augmentation layers default to *training* behaviour when a model is
    called without ``training=``. Always call with ``training=False`` or use ``predict_embeddings``.
    """
    return keras.Model(model.input, [model.get_layer(EMBEDDING).output, model.output], name="embed_and_score")


def predict_embeddings(model: keras.Model, images, batch_size: int = 64):
    """Inference-mode (embedding, p_referable) for a batch of 0-255 RGB images."""
    import numpy as np

    sub = embedding_model(model)
    embs, scores = [], []
    for i in range(0, len(images), batch_size):
        e, s = sub(np.asarray(images[i : i + batch_size], dtype="float32"), training=False)
        embs.append(np.asarray(e))
        scores.append(np.asarray(s).reshape(-1))
    return np.concatenate(embs), np.concatenate(scores)
