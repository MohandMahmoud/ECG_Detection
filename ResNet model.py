import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directories
train_data_dir = "C:/Users/Lenovo/PycharmProjects/Honda/train"
test_data_dir = "C:/Users/Lenovo/PycharmProjects/Honda/test"

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Data Preparation: Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Data Preparation: Testing Data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define Residual Block
def residual_block(x, filters, kernel_size=3, stride=1):
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)

    # Shortcut Connection
    if stride == 1:
        shortcut = x
    else:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([y, shortcut])
    y = layers.Activation('relu')(y)
    return y

# Model: ResNet
input_tensor = layers.Input(shape=(img_size[0], img_size[1], 3))
x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

for _ in range(3):
    x = residual_block(x, 64)

x = residual_block(x, 128, stride=2)
for _ in range(3):
    x = residual_block(x, 128)

x = residual_block(x, 256, stride=2)
for _ in range(3):
    x = residual_block(x, 256)

x = residual_block(x, 512, stride=2)
for _ in range(3):
    x = residual_block(x, 512)

x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(5, activation='softmax')(x)

model_resnet = models.Model(inputs=input_tensor, outputs=output)

# Compile ResNet model
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train ResNet
model_resnet.fit(train_generator, epochs=5, validation_data=test_generator)

# Clear TensorFlow Session to release memory
from tensorflow.keras import backend as K
K.clear_session()
