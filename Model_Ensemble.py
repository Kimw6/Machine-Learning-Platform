import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from utilityFunctions import train_model

BINARY = [True, False]

class RandomForestClassifierModel:
    def __init__(self):
        self.params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            # 'oob_score': False,
            # 'n_jobs': None,
            'random_state': None,
            # 'verbose': 0,
            # 'warm_start': False,
            # 'class_weight': None,
            # 'ccp_alpha': 0.0,
            # 'max_samples': None
        }
    

    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your Random Forest Classifier model.*** More 
                 info on [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)""")
        
        with st.form(key='random_forest_classifier_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input('n_estimators', value=100, step=1, min_value=1, key='n_estimators')
                st.selectbox('criterion', ['gini', 'entropy'], key='criterion')
                st.number_input('max_depth', min_value= 1, key='max_depth', value= None, step=1)
                st.number_input('min_samples_split', value=2, step=1, min_value=2, key='min_samples_split')
                st.number_input('min_samples_leaf', value=1, step=1, min_value=1, key='min_samples_leaf')
            
            with col2:
                st.number_input('min_weight_fraction_leaf', value=0.0, step=0.01, key='min_weight_fraction_leaf')
                st.selectbox('max_features', ['log2', 'sqrt'], key='max_features', index=1)
                st.number_input('max_leaf_nodes', min_value=2, key='max_leaf_nodes', step=1, value=None)
                st.number_input('min_impurity_decrease', value=0.0, step=0.01, key='min_impurity_decrease')
                st.selectbox('bootstrap', BINARY, key='bootstrap')
                
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        #st.write('Training Random Forest Classifier Model...')
        try:
            model = RandomForestClassifier(**self.params)
            train_model(model)
        except Exception as e:
            st.error(e)





class GradientBoostingClassifierModel:
    def __init__(self):
        self.params = {
            'loss': 'log_loss',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 1.0,
            'criterion': 'friedman_mse',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            # 'max_depth': 3,
            'min_impurity_decrease': 0.0,
            'init': None,
            'random_state': None,
            'max_features': None,
            'verbose': 0,
            'max_leaf_nodes': None,
            'warm_start': False,
            'validation_fraction': 0.1,
            'n_iter_no_change': None,
            'tol': 0.0001,
            'ccp_alpha': 0.0
        }
        st.session_state['init'] = None
        st.session_state['max_features'] = None
        st.session_state['verbose'] = 0
        st.session_state['tol'] = 0.0001
        st.session_state['ccp_alpha'] = 0.0

    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your Gradient Boosting Classifier model.*** More 
                 info on [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)""")
        
        with st.form(key='gradient_boosting_classifier_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox('loss', ['exponential', 'log_loss'], key='loss', index=1)
                st.number_input('learning_rate', value=0.1, step=0.01, min_value=0.01, max_value=1.0, key='learning_rate')
                st.number_input('n_estimators', value=100, step=1, min_value=1, key='n_estimators')
                st.number_input('subsample', value=1.0, step=0.01, min_value=0.01, max_value=1.0, key='subsample')
                st.number_input('min_impurity_decrease', value=0.0, step=0.01, key='min_impurity_decrease')
                st.selectbox('warm_start', BINARY, key='warm_start', index=1)
                st.number_input('n_iter_no_change', value=None, step=1, min_value=1, key='n_iter_no_change')
            
            with col2:
                st.selectbox('criterion', ['squared_error', 'friedman_mse'], key='criterion', index=1)
                st.number_input('min_samples_split', value=2, step=1, min_value=2, key='min_samples_split')
                st.number_input('min_samples_leaf', value=1, step=1, min_value=1, key='min_samples_leaf')
                st.number_input('min_weight_fraction_leaf', value=0.0, step=0.01, key='min_weight_fraction_leaf')
                st.number_input('max_leaf_nodes', min_value=2, key='max_leaf_nodes', step=1, value=None)
                st.number_input('validation_fraction', value=0.1, step=0.01, min_value=0.01, max_value=1.0, key='validation_fraction')
                
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        #st.write('Training Gradient Boosting Classifier Model...')
        try:
            model = GradientBoostingClassifier(**self.params)
            train_model(model)
        except Exception as e:
            st.error(e)





class AdaBoostClassifierModel:
    def __init__(self):
        self.params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R',
            'estimator': None
        }
        st.session_state['estimator'] = None

    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your AdaBoost Classifier model.*** More 
                 info on [AdaBoost Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)""")
        
        with st.form(key='adaboost_classifier_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input('n_estimators', value=50, step=1, min_value=1, key='n_estimators')
                st.number_input('learning_rate', value=1.0, step=0.01, min_value=0.01, max_value=10.0, key='learning_rate')
            
            with col2:
                st.selectbox('algorithm', ['SAMME', 'SAMME.R'], key='algorithm')
               
                
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        # if st.session_state['base_estimator']:
        #     self.params['base_estimator'] = eval(st.session_state['base_estimator'])
        
        #st.write('Training AdaBoost Classifier Model...')
        try:
            model = AdaBoostClassifier(**self.params)
            train_model(model)
        except Exception as e:
            st.error(e)

