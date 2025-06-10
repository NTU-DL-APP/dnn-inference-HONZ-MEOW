import tensorflow as tf
import numpy as np
import json
import os

# === 1. 載入訓練資料（使用 TensorFlow 內建 Fashion MNIST）===
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28 * 28)  # flatten

# === 2. 建立模型，只用 Dense / ReLU / Softmax ===
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# === 3. 編譯與訓練 ===
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# === 4. 儲存模型為 .h5 ===
model.save("model.h5")
print("✅ Saved model.h5")

# === 5. 轉換為 JSON（架構）與 NPZ（權重）格式 ===
# 5.1 儲存架構到 fashion_mnist.json
model_arch = []
for layer in model.layers:
    ltype = layer.__class__.__name__
    lname = layer.name
    activation = None
    if hasattr(layer, "activation"):
        activation = layer.activation.__name__
    config = {"activation": activation}
    weights_names = []
    if len(layer.get_weights()) == 2:
        weights_names = [f"{lname}_w", f"{lname}_b"]
    model_arch.append({
        "name": lname,
        "type": ltype,
        "config": config,
        "weights": weights_names
    })

with open("fashion_mnist.json", "w") as f:
    json.dump(model_arch, f, indent=2)
print("✅ Saved fashion_mnist.json")

# 5.2 儲存權重到 fashion_mnist.npz
weight_dict = {}
for layer in model.layers:
    if len(layer.get_weights()) == 2:
        w, b = layer.get_weights()
        weight_dict[f"{layer.name}_w"] = w
        weight_dict[f"{layer.name}_b"] = b

np.savez("fashion_mnist.npz", **weight_dict)
print("✅ Saved fashion_mnist.npz")
