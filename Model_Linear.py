import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from utilityFunctions import  train_model
import pandas as pd

BINARY = [True, False]



class LinearSupportVectorMachineModel:
    def __init__(self):

        self.params = {
            'penalty': 'l2',
            'loss': 'squared_hinge',
            'dual': 'auto',
            'tol': 0.0001,
            'C': 1.0,
            'multi_class': 'ovr',
            'fit_intercept': True,
            'intercept_scaling': 1,
            'class_weight': None,
            'verbose': 0,
            'random_state': None,
        }
 
 

    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your Linear Support Vector Machine (LinearSVC) model.*** More 
                 info on [Linear Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)""")
        
        with st.form(key='linear_support_vector_machine_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox('penalty', ['l1', 'l2'], key='penalty', index=1)
                st.selectbox('loss', ['hinge', 'squared_hinge'], key='loss')
                st.selectbox('dual', ['True', 'False', 'auto'], key='dual', index=2)
                st.number_input('tol', value=0.0001, step=0.0001, min_value=0.0001, max_value=1.0, key='tol')
                st.number_input('C', value=1.0, step=0.1, min_value=0.1, max_value=10.0, key='C')
            
            with col2:
                st.selectbox('multi_class', ['ovr', 'crammer_singer'], key='multi_class', index = 0)
                st.selectbox('fit_intercept', BINARY, key='fit_intercept', index=0)
                st.number_input('intercept_scaling', value=1.0, step=0.1, min_value=0.1, max_value=10.0, key='intercept_scaling')
                st.selectbox('class_weight', [None, 'balanced'], key='class_weight', index=0)
                st.number_input('verbose', value=0, step=1, min_value=0, max_value=1, key='verbose')
                
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        #st.write('Training Linear Support Vector Machine Model...')
        try :
            # st.write(st.session_state['df'])
            model = LinearSVC(**self.params)
            train_model(model)
        except Exception as e:
            st.error(e)
        
            


class LogisticRegressionModel:
    def __init__(self):
       

        self.params = {
            'penalty': 'l2',
            'C': 1.0,
            'intercept_scaling': 1,
            'max_iter': 100,
            'verbose': 0,
            'fit_intercept': True,
            'dual': False,
            'n_jobs': None,
            'l1_ratio': None,
            'tol': 0.0001,
            'class_weight': None,     
            'random_state': None,     
            'solver': 'lbfgs',      
            'warm_start': False, 
        }

    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your Logistic Regression model*** more 
                 info on [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)""")
        with st.form(key='logistic_regression_form'):
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox('penalty', ['l1', 'l2', 'elasticnet', 'none'], key='penalty', index=1)
                st.number_input('C', value=1.0, step=0.01, min_value=0.01, max_value=10.0, key='C')
                st.number_input('intercept_scaling', value=1.0, step=0.1, min_value=0.1, 
                                                     max_value=10.0, key='intercept_scaling')
                st.selectbox('multi_class', ['auto', 'ovr', 'multinomial'], index=0, key='multi_class')
                st.number_input('verbose', value=0, step=1, min_value=0, max_value=1, key='verbose')
                st.selectbox('class_weight', [None, 'balanced'], key='class_weight')  
                st.number_input('max_iter', value=100, step=10, min_value=10, max_value=1000, key='max_iter')
            with col2:
                st.selectbox('fit_intercept', BINARY, index=0, key = 'fit_intercept')
                st.selectbox('dual', BINARY, index=1, key='dual')
                st.number_input('n_jobs', value=None, step=1, min_value=1, max_value=10, key='n_jobs')
                st.number_input('l1_ratio', value=None, step=0.1, min_value=0.1, max_value=1.0, key='l1_ratio')
                st.number_input('tol', value=0.0001, step=0.0001, min_value=0.0001, max_value=0.01, key='tol')
                st.selectbox('solver', ['lbfgs', 'liblinear', 'sag', 'saga'], key='solver') 
                st.selectbox('warm_start', BINARY, key='warm_start')  
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        #st.write('Training Logistic Regression Model...')
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        try:
            model = LogisticRegression(**self.params)
            train_model(model)
        except Exception as e:
            st.error(e)
            
