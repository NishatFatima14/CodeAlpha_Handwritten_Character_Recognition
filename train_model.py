
import os
import numpy as np

# Safe imports for Keras/TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    # fallback to standalone keras (if someone installed that instead)
    import keras
    from keras import layers
    from keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf  # still import tf if available

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Config / hyperparameters
IMG_HEIGHT = 28
IMG_WIDTH = 28
CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10   # set to 10 for faster experiments; change to 25 for final run
SEED = 42
MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass

def load_mnist():
    # Use tf.keras.datasets to get MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()
print("Data shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Train/validation split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.12, random_state=SEED, stratify=y_train)
print("Train/Val/Test shapes:", x_train.shape, x_val.shape, x_test.shape)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.06,
    zoom_range=0.08
)
train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

# Residual block and model builder
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='resnet_mnist')
    return model

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"),
                                                save_best_only=True,
                                                monitor="val_accuracy", mode="max")
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

steps_per_epoch = max(1, len(x_train) // BATCH_SIZE)
history = model.fit(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint_cb, reduce_lr, early_stop],
                    verbose=2)

# Save final model (both .h5 and .keras recommended)
final_h5 = os.path.join(MODEL_DIR, "final_model.h5")
final_keras = os.path.join(MODEL_DIR, "final_model.keras")
model.save(final_h5)
try:
    model.save(final_keras)
except Exception:
    pass

# Evaluation and artifacts
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc*100:.2f}%, loss: {test_loss:.4f}")

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Simple README artifact
with open('README.txt', 'w') as f:
    f.write("Artifacts: saved_model folder with best_model.h5 and final_model.*.\\nconfusion_matrix.png saved.\\n")

print("Training script finished. Models saved in 'saved_model' directory.")