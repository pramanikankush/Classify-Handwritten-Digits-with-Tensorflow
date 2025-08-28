# 🖊️ MNIST Handwritten Digit Classification with TensorFlow

This repository demonstrates two approaches to classify handwritten digits (MNIST dataset):

1. **Linear Classifier (Logistic/Softmax Regression)**
2. **Deep Neural Network (Fully Connected Layers)**

---

## 📦 Installation
```bash
pip install tensorflow numpy matplotlib
🚀 Approach 1: Linear Classifier
A simple softmax regression model with one dense layer.

python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)).astype("float32") / 255.0
x_test  = x_test.reshape((-1, 28*28)).astype("float32") / 255.0

# Linear model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28*28,)),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"Linear Classifier Test Accuracy: {acc:.4f}")
✅ Accuracy: ~91%

🚀 Approach 2: Deep Neural Network (DNN)
A feed-forward neural net with 2 hidden layers.

python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print(f"DNN Test Accuracy: {acc:.4f}")
✅ Accuracy: ~97%

📊 Results
Model	Test Accuracy
Linear Classifier	~91%
DNN (2 hidden)	~97%

💡 Next Steps
Use CNN (Conv2D + Pooling) to push accuracy >99%.

Try data augmentation for robustness.

Deploy model using TensorFlow Serving or Flask.

📂 Saving & Reloading Model
python
Copy code
# Save
model.save("epic_num_reader.h5")

# Reload
new_model = tf.keras.models.load_model("epic_num_reader.h5")
print("Reloaded Model Accuracy:", new_model.evaluate(x_test, y_test, verbose=0)[1])
📸 Example Prediction
python
Copy code
idx = 0
pred = np.argmax(model.predict(x_test[idx:idx+1]), axis=1)[0]
plt.imshow(x_test[idx].reshape(28,28), cmap="gray")
plt.title(f"Label:{y_test[idx]} | Pred:{pred}")
plt.show()
🔮 Future Improvements
Convolutional Neural Networks (CNNs)

Batch Normalization & Dropout

Hyperparameter tuning with KerasTuner

Model deployment (Streamlit / Flask)

👨‍💻 Author: Ankush
