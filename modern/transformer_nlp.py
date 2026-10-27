"""Minimal Keras Transformer encoder block for sequence classification experiments.
This is a teaching scaffold: supply integer token sequences and labels from your own dataset.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False, mask=None):
        attention_mask = None
        if mask is not None:
            attention_mask = tf.cast(mask[:, tf.newaxis, :], tf.bool)
        attn = self.attention(inputs, inputs, attention_mask=attention_mask, training=training)
        x = self.norm1(inputs + self.drop1(attn, training=training))
        ffn = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(ffn, training=training))

def build_text_classifier(vocab_size=20000, max_length=200, embed_dim=128, num_heads=4, ff_dim=256):
    tokens = keras.Input(shape=(max_length,), dtype="int32", name="token_ids")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(tokens)
    positions = tf.range(start=0, limit=max_length, delta=1)
    position_embedding = layers.Embedding(max_length, embed_dim)(positions)
    x = x + position_embedding
    x = TransformerEncoder(embed_dim, num_heads, ff_dim, name="transformer_encoder")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid", name="sentiment_probability")(x)
    model = keras.Model(tokens, output, name="transformer_text_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

if __name__ == "__main__":
    model = build_text_classifier()
    model.summary()
    print("Provide padded integer token sequences and binary labels to train this model.")
