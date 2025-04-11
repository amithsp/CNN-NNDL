import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Original dataset folders
ai_art_dir = r"C:\Users\amith\Downloads\AiArtData\AiArtData"
real_art_dir = r"C:\Users\amith\Downloads\RealArt\RealArt"

# Temporary train/test split folder
base_dir = 'temp_dataset_split'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create split folders
def create_split_folders():
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    os.makedirs(os.path.join(train_dir, 'ai_art'))
    os.makedirs(os.path.join(train_dir, 'real_art'))
    os.makedirs(os.path.join(test_dir, 'ai_art'))
    os.makedirs(os.path.join(test_dir, 'real_art'))

    def split_and_copy(class_name, source_dir):
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

        for f in train_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, class_name, f))
        for f in test_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, class_name, f))

    split_and_copy('ai_art', ai_art_dir)
    split_and_copy('real_art', real_art_dir)

create_split_folders()

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define AlexNet model
def build_alexnet(input_shape=(224, 224, 3), num_classes=1):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

        tf.keras.layers.Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

        tf.keras.layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

# Build and compile the model
model = build_alexnet()
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# Save the trained model
model.save('ai_vs_real_art_alexnet_model.h5')

# Plot training history
def plot_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)
