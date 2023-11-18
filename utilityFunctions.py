
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance

def train_model(model, feature_importance=False, decision_tree=False):
    df = st.session_state['df']
    target = st.session_state['target']
    test_size = st.session_state['test_size']
    random_state = st.session_state['random_state']
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    if feature_importance:
        st.write('Feature Importance')
    if decision_tree:
        permutation_importance(model, X_test, y_test)
    #     render_tree(model, X, y)
    return (model, y_pred)


# This need needs to be fixed. 
def render_tree(model, columns, target):
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a Matplotlib figure and axis
    plot_tree(model, feature_names=columns, class_names=[str(t) for t in target], filled=True, ax=ax)
    st.pyplot(fig)

 


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write('***Accuracy: {:.2f}***'.format(accuracy))
    st.write('Classification Report')
    metrics_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    data = pd.DataFrame(metrics_report).transpose()
    st.write(data)
    # for row in metrics_report:
    #     st.write(row, metrics_report[row])
    confusion_matrix(y_test, y_pred)
    return y_pred


def confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    
    # Render the confusion matrix using heatmap
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Pass the Matplotlib figure to st.pyplot()
    st.pyplot(fig)
    
    return cm


def perm_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, random_state=42)
    st.write(result.importances_mean)