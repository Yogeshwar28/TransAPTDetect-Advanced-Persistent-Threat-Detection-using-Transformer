import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from sklearn.metrics import balanced_accuracy_score

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
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

test_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans/Unraveled_test_data_standard_scaled.csv"
test_df = pd.read_csv(test_path)

X_test = test_df.drop(columns=["Stage"]).values
y_true = test_df["Stage"].values

with open("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

y_test = label_encoder.transform(y_true)
class_names = label_encoder.classes_

model_path = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/final_transformer_model.keras"
model = load_model(model_path, custom_objects={"TransformerBlock": TransformerBlock})

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

y_pred = model.predict(X_test, verbose=1).argmax(axis=1)

bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"\Balanced Accuracy: {bal_acc*100:.2f}%")

print("\Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/confusion_matrix.png")
plt.show()

results = pd.DataFrame(X_test)
results["True_Label"] = y_test
results["True_Stage"] = label_encoder.inverse_transform(y_test)
results["Pred_Label"] = y_pred
results["Pred_Stage"] = label_encoder.inverse_transform(y_pred)

output_csv = "/home/yogeshwar/Yogesh-MTP/Unravled_Dataset/unraveled/data/Combined_New/New/Trans_Updated2/Improved/transformer_predictions.csv"
results.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
