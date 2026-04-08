import tensorflow as tf 
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout 
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from tensorflow.keras.optimizers import Adam 
IMG_SIZE = 224 
BATCH_SIZE = 32 
train_datagen = ImageDataGenerator( 
    preprocessing_function=preprocess_input,
    rotation_range=20, 
    zoom_range=0.2, 
    horizontal_flip=True ) 

val_datagen = ImageDataGenerator( 
    preprocessing_function=preprocess_input )

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=BATCH_SIZE, 
    class_mode="categorical" 
    )

val_generator = val_datagen.flow_from_directory( 
    "data/val", 
    target_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=BATCH_SIZE,
    class_mode="categorical" 
    )

base_model = EfficientNetB0( 
    weights="imagenet", 
    include_top=False,
    input_shape=(224, 224, 3) 
    ) 
base_model.trainable = False
x = base_model.output 
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x) 
x = Dense(128, activation="relu")(x) 
output = Dense(4, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)


model.compile( 
              optimizer=Adam(learning_rate=0.0001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"]
    )

callbacks = [ 
    EarlyStopping(patience=5, restore_best_weights=True), 
    ModelCheckpoint("best_model.keras", 
save_best_only=True) 
] 

history = model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=50, 
    callbacks=callbacks
    ) 
model.save("dermalscan_model.keras")