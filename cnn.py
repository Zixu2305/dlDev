import tensorflow as tf
from tensorflow.keras import Sequential, utils, layers, optimizers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
import json
import datetime
import math
from tensorflow.keras.mixed_precision import set_global_policy

# Enable GPU memory growth and set memory limit
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)]
        )
        print("Memory growth enabled with 6.5 GB limit.")
    except RuntimeError as e:
        print(f"Error: {e}")

# Set the global mixed precision policy to 'mixed_float16'
set_global_policy('mixed_float16')

# Hyperparameters
imgSize = 224
batchSize = 16
basePath = os.getcwd()
dataPath = os.path.join(basePath, 'ASL_Dataset/Train')

# Callbacks
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
model_checkpoint = ModelCheckpoint(
    filepath=basePath + '/checkpoints/model_epoch-{epoch:02d}_val_loss-{val_loss:.4f}.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
callbacks = [tensorboard_callback, early_stopping, reduce_lr, model_checkpoint]

# Load the dataset
AUTOTUNE = tf.data.AUTOTUNE
dsTrain, dsVal = utils.image_dataset_from_directory(
    dataPath,
    image_size=(imgSize, imgSize),
    batch_size=batchSize,
    label_mode='categorical',
    validation_split=0.2,
    subset='both',
    seed=123
)

# Save class names
class_names = dsTrain.class_names
with open('labels.json', 'w') as f:
    json.dump(class_names, f)
steps_per_epoch = math.ceil(178460 / batchSize)  # Adjust based on your dataset size
validation_steps = math.ceil(44614 / batchSize)  # Adjust as needed

# Data augmentation and preprocessing
def preprocess(image, label):
    image = tf.cast(image, tf.float32)  # Ensure float32
    image = preprocess_input(image)    # Use InceptionV3's preprocessing
    return image, label

augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomContrast(0.2),
])

dsTrain = (
    dsTrain.map(lambda x, y: (augmentation(x), y), num_parallel_calls=AUTOTUNE)
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=100)
    .repeat()
    .prefetch(buffer_size=AUTOTUNE)
)

dsVal = dsVal.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

# Pretrained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(imgSize, imgSize, 3))

# Unfreeze the top layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:150]:  # Freeze the first 150 layers
    layer.trainable = False

# Model architecture
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=L2(0.02)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=L2(0.02)),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax'),
])

# Optimizer with mixed precision
base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=base_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    dsTrain,
    validation_data=dsVal,
    epochs=50,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)
