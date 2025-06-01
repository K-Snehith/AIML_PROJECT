import os
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

train_path = 'asl_alphabet_train'
img_size = 64  # Resize images to 64x64
data = []
labels = []

classes = sorted(os.listdir(train_path))  # A-Z
print("Classes found:", classes)

# Load images and labels
for class_idx, label in enumerate(classes):
    folder = os.path.join(train_path, label)
    if not os.path.isdir(folder):
        continue
    count = 0
    for file in os.listdir(folder):
        if count >= 500:  # limit to 500 images per class
            break
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(class_idx)
        count += 1

data = np.array(data) / 255.0
labels = to_categorical(labels)
print("Data loaded:", data.shape)

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Save model and labels
model.save('sign_language_model.h5')
np.save('classes.npy', np.array(classes))
print("Model and classes saved.")

