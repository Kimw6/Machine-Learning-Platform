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
    st.write(X_train.shape[1])
    

    units_per_layer = []
    total = 16*num_hidden_layers
    if total < 128:
        total = 128

    model.add(Dense(units=total, activation='relu', input_dim=X_train.shape[1], name='input_layer'))
    for _ in range(num_hidden_layers):
        total = total//2
        if total < 16:
            total = 16
        units_per_layer.append(total)
    for layer, unit in enumerate(units_per_layer):
        name = 'hidden_layer_{}'.format(layer)
        model.add(Dense(units=unit, activation='relu', name=name))

    num_classes = len(np.unique(y_train))
    activation = 'sigmoid' if num_classes == 2 else 'softmax'
    model.add(Dense(units=num_classes, activation=activation))
    model.compile(optimizer='adam', loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy', metrics=['accuracy'])
    return model
