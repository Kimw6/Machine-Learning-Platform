import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

BINARY = [True, False]
class KMeansClusteringModel:
    def __init__(self):
        self.params = {
            'n_clusters': 8,
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
        st.write("""***Please adjust the parameters below to configure your K-Means clustering model.*** More 
                 info on [K-Means Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)""")
        
        with st.form(key='kmeans_clustering_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input('n_clusters', value=8, step=1, min_value=1, key='n_clusters')
                st.selectbox('init', ['k-means++', 'random'], key='init')
                st.number_input('n_init', value=10, step=1, min_value=1, key='n_init')
                st.number_input('max_iter', value=300, step=10, min_value=1, key='max_iter')
                st.selectbox('copy_x', BINARY, key='copy_x', index=0)
            
            with col2:
                st.number_input('tol', value=0.0001, step=0.0001, min_value=0.0, key='tol')
                st.selectbox('algorithm', ['lloyd', 'elkan'], key='algorithm')
                st.text("Optional: You can set a random seed for reproducibility:")
                st.number_input('random_state', min_value=1, key='random_state')
                
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
      
        
        st.write('Training K-Means Clustering Model...')
        try:
            model = KMeans(**self.params)
            train_model(model)
        except Exception as e:
            st.error(e)



def train_model(kmeans):
    X = st.session_state['df'].drop(st.session_state['target'], axis=1)

    st.title('K-Means Clustering Visualization')
    kmeans.fit(X)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    fig, ax = plt.subplots()
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=50, cmap='viridis')
    
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=200, c='red', label='Cluster Centers')
    st.write(ax)
    st.pyplot(fig)

