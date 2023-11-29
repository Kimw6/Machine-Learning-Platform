import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from utilityFunctions import confusion_matrix, handle_errors
import matplotlib.pyplot as plt



def create_model(df, target, num_hidden_layers, random_state):

    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=random_state)

    # Create a Sequential model
    model = Sequential()
    

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

def train_model(model, epochs, random_state):
    df = st.session_state['df']
    target = st.session_state['target']
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    loss = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
    results = model.evaluate(X_test, y_test)
    preds = model.predict(X_test)
    y_pred = []
    for pred in preds:
        if pred[0] > pred[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    # confusion_matrix(y_test, y_pred)
    plot_loss(loss)
    plot_loss(loss, type='accuracy')

def plot_loss(loss, type='loss'):
    if type == 'loss':
        training_loss = loss.history['loss']
        validation_loss = loss.history['val_loss']
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss', marker='o')
        plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
    if type == 'accuracy':
        training_loss = loss.history['accuracy']
        validation_loss = loss.history['val_accuracy']
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Accuracy', marker='o')
        plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Accuracy Over Epochs')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
    




