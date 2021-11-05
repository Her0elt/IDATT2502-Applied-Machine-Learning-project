
from typing import Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import  RMSprop
from tensorflow.keras.models import load_model

import numpy as np


class DQN:
    def __init__(self, input_shape: tuple, action_shape: int):
        self.model = Sequential()
        self.model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', kernel_initializer='random_uniform'))
        self.model.add(Dense(action_shape, activation="softmax"))   
        self.model.compile(loss="mse", optimizer=RMSprop(
            learning_rate=0.00025), metrics=["accuracy"])
    
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, verbose: 0 | 1):
        self.model.fit(x, y, batch_size=batch_size, verbose=verbose)
    
    def predict(self, x: np.ndarray) -> Any:
        return self.model.predict(x)

    def save_model(self, file_name: str):
        self.model.save(file_name)

    def load_model(self, file_name: str):
        self.model = load_model(file_name)
        