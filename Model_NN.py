import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def create_and_train_nn(df, target, num_hidden_layers, random_state):

    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=random_state)

    # Create a Sequential model
    model = Sequential()


    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))


    # for units in units_per_layer[:num_hidden_layers]:
    #     model.add(Dense(units=units, activation='relu'))
    for _ in range(num_hidden_layers):
        model.add(Dense(units=64, activation='relu'))

    # Add output layer
    num_classes = len(np.unique(y_train))
    activation = 'sigmoid' if num_classes == 2 else 'softmax'
    model.add(Dense(units=num_classes, activation=activation))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    #model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Return the trained model
    return model
