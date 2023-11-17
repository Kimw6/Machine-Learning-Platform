import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

class PCAModel:
    def __init__(self):
        self.params = {
            'n_components': None,
            'copy': True,
            'whiten': False,
            'svd_solver': 'auto',
            'tol': 0.0,
            'iterated_power': 'auto',
            'n_oversamples': 10,
            'power_iteration_normalizer': 'auto',
            'random_state': None,
        }
    
    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your PCA model.*** More 
                 info on [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)""")
        
        with st.form(key='pca_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input('n_components', min_value=1, value = 2,  max_value=30, key='n_components')
                st.selectbox('svd_solver', ['auto', 'full', 'arpack', 'randomized'], key='svd_solver')
                st.number_input('tol', step=0.01, key='tol')
                st.selectbox('copy', [True, False], key='copy', index=0)
                st.number_input('n_oversamples', min_value=1, key='n_oversamples')
            
            with col2:
                
                st.selectbox('iterated_power', ['auto', 'exact', 'randomized'], key='iterated_power')
                st.selectbox('power_iteration_normalizer', ['auto', 'QR', 'LU'], key='power_iteration_normalizer')
                st.number_input('random_state', min_value=1, value = 42, key='random_state')
                st.checkbox('Whiten', key='whiten')
                
            st.form_submit_button('Apply PCA', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
      
        try:
            pca = PCA(**self.params)
            self.train_model(pca)
        except Exception as e:
            st.error(e)

    def train_model(self, pca):
        X = st.session_state['df'].drop(st.session_state['target'], axis=1)

        # Apply PCA on the example data
        pca.fit(X)
        components= pca.transform(X)

        st.write("***Original shape:*** ", X.shape)
        st.write("***Transformed shape:***", components.shape)
 
        df = pd.DataFrame(components)
        target = st.session_state['target']
        df[target] = st.session_state['df'][target]
        st.write("Below is the few examples of the transformed data:")
        st.write(df.head(10))
        st.write("You can download the transformed data by clicking the button below:")
        st.download_button('Download Data', data=df.to_csv(index=False), file_name='transformed_data.csv', mime='text/csv')
       

        

        
        

  