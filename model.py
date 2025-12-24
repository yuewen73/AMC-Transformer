# model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# -------------------------
# Custom Layers
# -------------------------

@register_keras_serializable(package="custom")
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 2, self.patch_size, 1],
            strides=[1, 2, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


@register_keras_serializable(package="custom")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = int(num_patches)
        self.projection_dim = int(projection_dim)
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return cfg


# -------------------------
# Model Builder
# -------------------------

def create_vit_classifier(
    input_shape=(2, 1024, 1),
    patch_size=16,
    projection_dim=96,
    num_heads=4,
    transformer_layers=10,
    transformer_units=None,
    mlp_head_units=None,
    num_classes=24,
    learning_rate=1e-3,
):
    if transformer_units is None:
        transformer_units = [projection_dim * 2, projection_dim]
    if mlp_head_units is None:
        mlp_head_units = [2048, 1024]

    num_patches = input_shape[1] // patch_size

    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    x = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attn, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp_block(x3, transformer_units, 0.1)
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = mlp_block(x, mlp_head_units, 0.5)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def mlp_block(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
