import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import time



# Load Data
def load_and_split_data():
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # CIFAR-10 Class Names
    CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck']
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    x_train = x_train_full[:40000].astype('float32')
    y_train = y_train_full[:40000]

    x_val = x_train_full[40000:].astype('float32')
    y_val = y_train_full[40000:]

    x_test = x_test.astype('float32')

    return x_train, y_train, x_val, y_val, x_test, y_test


def train_and_evaluate(model, x_tr, y_tr, x_v, y_v, x_te, y_te,
epochs=20, batch_size=128, extra_callbacks=None, aug=False):
    cb = extra_callbacks if extra_callbacks else []
    start = time.time()
    history = model.fit(x_tr, to_categorical(y_tr, 10),
    validation_data=(x_v, to_categorical(y_v, 10)),
    epochs=epochs, batch_size=batch_size,
    callbacks=cb, verbose=0) if not aug else model.fit(x_tr,
        validation_data=(x_val_C, to_categorical(y_val,10)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb, verbose=0
    )

    elapsed = time.time() - start
    test_loss, test_acc = model.evaluate(x_te, to_categorical(y_te, 10), verbose=0)
    print(f"Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f} | Time:{elapsed:.1f}s")
    return history, test_acc, test_loss, elapsed


# build baseline model for CNN
def BaselineCNN():
    model = models.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Conv2D(32,(3,3), activation= "relu", padding= "same"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation= "relu", padding= "same"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation= "relu"),
        layers.Dense(10, activation= "softmax")
    ])
    return model

# Experiment C
def run_Experiment_C():
    print("Experiment C: Standardization per-channel")

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_split_data()

    # Standardization
    mean = np.mean(x_train, axis=(0, 1, 2))
    std = np.std(x_train, axis=(0, 1, 2))

    x_train_C = (x_train - mean) / std
    x_val_C = (x_val - mean) / std
    x_test_C = (x_test - mean) / std

    # Build model
    model_C = BaselineCNN()
    model_C.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train & Evaluate
    history, acc, loss, t = train_and_evaluate(
        model_C,   
        x_train_C, y_train,
        x_val_C, y_val,
        x_test_C, y_test
    )

    return history, acc, loss, t