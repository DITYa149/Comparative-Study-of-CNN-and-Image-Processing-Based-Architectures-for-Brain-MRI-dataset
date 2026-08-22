import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class BrainMRICNN:

    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):

        inputs = keras.Input(shape=self.input_shape, name='mri_input')

        # ===== CNN BASE =====
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # ✅ IMPROVED
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # ===== OUTPUT HEADS =====

        # Tumor
        tumor_branch = layers.Dense(128, activation='relu')(x)
        tumor_output = layers.Dense(5, activation='softmax', name='brain_tumor')(tumor_branch)

        # Stroke
        stroke_branch = layers.Dense(64, activation='relu')(x)
        stroke_output = layers.Dense(2, activation='softmax', name='stroke')(stroke_branch)

        # MS
        ms_branch = layers.Dense(64, activation='relu')(x)
        ms_output = layers.Dense(2, activation='softmax', name='ms')(ms_branch)

        self.model = models.Model(
            inputs=inputs,
            outputs=[tumor_output, stroke_output, ms_output],
            name='Brain_MRI_CNN'
        )

        return self.model

    def compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss={
                'brain_tumor': 'categorical_crossentropy',
                'stroke': 'categorical_crossentropy',
                'ms': 'categorical_crossentropy'
            },
            loss_weights={
                'brain_tumor': 1.0,
                'stroke': 0.8,
                'ms': 0.8
            },
            metrics={
                'brain_tumor': ['accuracy'],
                'stroke': ['accuracy'],
                'ms': ['accuracy']
            }
       )