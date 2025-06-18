import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,        # Normalize pixel values to [0, 1]
    shear_range=0.2,        # Random shear transformations
    zoom_range=0.2,         # Random zoom
    horizontal_flip=True    # Random horizontal flips
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255         # Normalize pixel values to [0, 1]
)

# Load training data
train_data = train_datagen.flow_from_directory(
    'data/train',           # Path to training data
    target_size=(128, 128), # Resize images to 128x128
    batch_size=32,          # Batch size
    class_mode='binary'     # Binary classification (benign vs malignant)
)

# Load testing data
test_data = test_datagen.flow_from_directory(
    'data/test',            # Path to testing data
    target_size=(128, 128), # Resize images to 128x128
    batch_size=32,          # Batch size
    class_mode='binary'     # Binary classification
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),           # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Adam optimizer with low learning rate
    loss='binary_crossentropy',            # Binary classification loss
    metrics=['accuracy']                   # Track accuracy
)

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // 32,  # Number of batches per epoch
    epochs=10,                                 # Number of epochs
    validation_data=test_data,
    validation_steps=test_data.samples // 32   # Validation batches
)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Save the trained model
model.save('weights.h5')
print("Model saved as weights.h5")