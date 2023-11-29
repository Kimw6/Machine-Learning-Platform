import streamlit as st
from sklearn.cluster import KMeans
from utilityFunctions import evaluate_model
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BINARY = [True, False]
class KMeansClusteringModel:
    def __init__(self):
        self.params = {
            'n_clusters': 1,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'tol': 0.0001,
            'random_state': None,
            'copy_x': True,
            'algorithm': 'lloyd',
            'random_state': None,
        }
    

    def parameters(self):
        st.write("""Please adjust the parameters below to configure your K-Means clustering model. 
                 ***Note that number of clusters set to number of classes avaible in target.***
                More info on [K-Means Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)""")
        num_clusters = len(st.session_state['df'][st.session_state['target']].unique())
        with st.form(key='kmeans_clustering_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input('n_clusters', value=num_clusters, step=1, min_value=1, key='n_clusters')
                st.selectbox('init', ['k-means++', 'random'], key='init')
                st.number_input('n_init', value=10, step=1, min_value=1, key='n_init')
                st.number_input('max_iter', value=300, step=10, min_value=1, key='max_iter')
                st.selectbox('copy_x', BINARY, key='copy_x', index=0)
            
            with col2:
                st.number_input('tol', value=0.0001, step=0.0001, min_value=0.0, key='tol')
                st.selectbox('algorithm', ['lloyd', 'elkan'], key='algorithm')
                #st.text("Optional: You can set a random seed for reproducibility:")
                st.number_input('random_state', min_value=1, value=42, key='random_state')
                
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
      
        
        try:
            model = KMeans(**self.params)
            self.train_model(model)
        except Exception as e:
            st.error(e)





    def train_model(self,kmeans):
        X = st.session_state['df'].drop(st.session_state['target'], axis=1)
        st.title('K-Means Clustering Visualization')
        
        kmeans.fit(X)
        labels = kmeans.labels_
        X1 = st.session_state['df']
        X1['Cluster labels'] = labels
        target = np.array(st.session_state['df'][st.session_state['target']])

        num_classes = len(st.session_state['df'][st.session_state['target']].unique())
        if num_classes == self.params['n_clusters']:
            metrics_report = metrics.classification_report(target, labels, output_dict=True)
            data = pd.DataFrame(metrics_report).transpose()
            st.write(data)
            self.confusion_matrix(target, labels)
        st.write('Data with Cluster Labels added')
        st.write(X1)
        st.download_button('Download Data', data=X.to_csv(index=False), file_name='clustered_data.csv', mime='text/csv')

    def confusion_matrix(self, y_test, y_pred):
        cm = metrics.confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        

