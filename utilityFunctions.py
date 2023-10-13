
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn import metrics
from sklearn.tree import export_graphviz
import graphviz

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

        render_tree(model, X.columns, y.unique())
    return (model, y_pred)

def render_tree(model, columns, target):
    data = export_graphviz(model, out_file=None, 
                           feature_names=columns, 
                           class_names=[str(t) for t in target],  
                           rounded=True,
                           filled=True, 
                           special_characters=True)
    tree = graphviz.Source(data)
    st.write(tree)
 


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    st.write('Accuracy: {:.2f}'.format(accuracy))
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