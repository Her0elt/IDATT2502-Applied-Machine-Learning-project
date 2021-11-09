from typing import Any

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop

from constants import LEARNING_RATE


class DQN:
    def __init__(self, input_shape: tuple, action_shape: int):

        self.model = Sequential(
            [   
                Conv2D(
                    filters=32,
                    kernel_size=(8, 8),
                    strides=(4, 4),
                    activation="relu",
                    input_shape=input_shape,
                ),
                Conv2D(
                    filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
                ),
                Conv2D(
                    filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu"
                ),
                Flatten(),
                Dense(128, activation="relu", kernel_initializer="he_uniform"),
                Dense(64, activation="relu", kernel_initializer="he_uniform"),
                Dense(32, activation="relu", kernel_initializer="he_uniform"),
                Dense(16, activation="relu", kernel_initializer="he_uniform"),
                Dense(action_shape, activation="softmax"),
            ]
        )
        self.model.compile(
            loss="mse",
            optimizer=RMSprop(learning_rate=LEARNING_RATE),
            metrics=["accuracy"],
        )

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, verbose: 0 | 1):
        self.model.fit(x, y, batch_size=batch_size, verbose=verbose)

    def predict(self, x: np.ndarray) -> Any:
        return self.model.predict(x)

    def save_model(self, file_name: str):
        self.model.save(file_name)

    def load_model(self, file_name: str):
        self.model = load_model(file_name)
