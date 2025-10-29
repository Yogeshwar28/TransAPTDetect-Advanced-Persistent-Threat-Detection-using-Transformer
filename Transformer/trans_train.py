import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle


strategy = tf.distribute.MirroredStrategy()
print("GPUs being used:", strategy.num_replicas_in_sync)

train_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/Unraveled_train_split.csv"
val_path   = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/Unraveled_val_split.csv"

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)

X_train = train_df.drop(columns=["Stage"]).values
y_train = train_df["Stage"].values

X_val   = val_df.drop(columns=["Stage"]).values
y_val   = val_df["Stage"].values

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded   = label_encoder.transform(y_val)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weights = dict(zip(np.unique(y_train_encoded), class_weights_array))
print("Class Weights:", class_weights)

def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads, dtype=tf.float32)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

def create_transformer_model(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)

    embed_dim = 64
    x = Dense(embed_dim)(x)
    x = x + positional_encoding(input_dim, embed_dim)

    x = TransformerBlock(embed_dim=embed_dim, num_heads=4, ff_dim=128)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)

class MacroF1Callback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        f1 = f1_score(self.y_val, y_pred, average="macro")
        logs["val_macro_f1"] = f1
        print(f" — val_macro_f1: {f1:.4f}")

checkpoint_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/best_transformer.keras"
callbacks = [
    MacroF1Callback(X_val, y_val_encoded),
    EarlyStopping(monitor="val_macro_f1", mode="max", patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor="val_macro_f1", mode="max", save_best_only=True)
]

with strategy.scope():
    model = create_transformer_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_train_encoded)))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train_encoded,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val_encoded),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

model.save("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/final_transformer_model.keras")
print("\Model saved successfully")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
if "val_macro_f1" in history.history:
    plt.plot(history.history["val_macro_f1"], label="Val Macro-F1")
plt.title("Loss & F1 Curve")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/training_curves.png")
