import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from utilityFunctions import train_model

# Constants
BINARY = [True, False]

class DecisionTreeClassifierModel:
    def __init__(self):
        self.params = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': None,
            'random_state': None,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'class_weight': None,
            'ccp_alpha': 0.0,
        }
        for key in self.params.keys():
            if key not in st.session_state:
                st.session_state[key] = self.params[key]

    def parameters(self):
        st.write("""***Please adjust the parameters below to configure your Decision Tree Classifier model*** more 
                 info on [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)""")
        with st.form(key='decision_tree_classifier_form'):
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox('criterion', ['gini', 'entropy'], key='criterion')
                st.selectbox('splitter', ['best', 'random'], key='splitter')
                st.number_input('max_depth', min_value=1, key='max_depth')
                st.number_input('min_samples_split', min_value=2, key='min_samples_split')
                st.number_input('min_samples_leaf', min_value=1, key='min_samples_leaf')
            with col2:
                st.number_input('min_weight_fraction_leaf', step=0.01, key='min_weight_fraction_leaf')
                st.selectbox('max_features', [None, 'auto', 'sqrt', 'log2'], key='max_features')
                st.number_input('max_leaf_nodes', value=2, max_value=1000,  step= 1, key='max_leaf_nodes')
                st.number_input('min_impurity_decrease', step=0.01, key='min_impurity_decrease')
                st.selectbox('class_weight', [None, 'balanced'], key='class_weight')
                st.number_input('ccp_alpha', step=0.01, key='ccp_alpha')
            st.form_submit_button('Train Model', on_click=self.on_click)

    def on_click(self):
        st.write('Training Decision Tree Classifier Model...')
        self.params = {key: st.session_state[key] for key in self.params.keys()}
        try:
            model = DecisionTreeClassifier(**self.params)
            train_model(model, decision_tree=True)
            
        except Exception as e:
            st.error(e)


