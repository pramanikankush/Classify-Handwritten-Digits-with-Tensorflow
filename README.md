# MNIST Handwritten Digit Classification with TensorFlow

This repo shows two approaches to classify handwritten digits (MNIST):

1. **Linear Classifier (Logistic Regression)**
2. **Deep Neural Network (Fully Connected Layers)**

---

## Install Dependencies
```bash
pip install tensorflow numpy matplotlib
1️⃣ Linear Classifier (Softmax Regression)
python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)).astype("float32") / 255.0
x_test  = x_test.reshape((-1, 28*28)).astype("float32") / 255.0

# Model: single Dense layer = linear classifier
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

# Predict example
idx = 0
pred = np.argmax(model.predict(x_test[idx:idx+1]), axis=1)[0]
print("Label:", y_test[idx], "Prediction:", pred)

plt.imshow(x_test[idx].reshape(28,28), cmap="gray")
plt.title(f"Label:{y_test[idx]} | Pred:{pred}")
plt.show()
2️⃣ Deep Neural Network (DNN with 2 hidden layers)
python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Model: Flatten + 2 Dense hidden layers + Output
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

# Predict example
idx = 2
pred = np.argmax(model.predict(x_test[idx:idx+1]), axis=1)[0]
print("Label:", y_test[idx], "Prediction:", pred)

plt.imshow(x_test[idx], cmap="gray")
plt.title(f"Label:{y_test[idx]} | Pred:{pred}")
plt.show()

# Save and reload model
model.save("epic_num_reader.h5")
new_model = tf.keras.models.load_model("epic_num_reader.h5")
print("Reloaded Model Accuracy:", new_model.evaluate(x_test, y_test, verbose=0)[1])
Results
Linear Classifier: ~91% accuracy

Deep Neural Network: ~97% accuracy

For even higher accuracy (>99%), consider using a CNN (Conv2D + Pooling).
