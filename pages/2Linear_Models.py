import streamlit as st
from Model_Linear import LinearSupportVectorMachineModel, LogisticRegressionModel




# Importing the necessary libraries
import streamlit as st
from Model_Linear import LinearSupportVectorMachineModel, LogisticRegressionModel

# Displaying a subheader for the linear models for classification
st.subheader('`Linear Models for Classification`')

# Defining the default session state
default_session_state = {
    'df': None,
    'target': None,
    'linear_model': None,
    'test_size': None,
    'random_state': None,
    'train' : None,
}

# Initializing the session state with default values
for key, value in default_session_state.items():
    if key not in st.session_state:       
        st.session_state[key] = value

# List of available model types
model_type = ['Linear Support Vector Machine', 'Logistic Regression']

# Defining the linear_model function
def linear_model():

    # Function to be executed when the submit button is clicked
    def on_click():
        if st.session_state['temp_linear_model'] is None:
            st.warning('Please select a model.', icon="⚠️")
            return
        if st.session_state['temp_test_size'] is None:
            st.warning('Please select a test size.', icon="⚠️")
            return
        if st.session_state['rand_state'] is None:
            st.warning('Please select a random state.', icon="⚠️")
            return
        st.session_state['linear_model'] = st.session_state['temp_linear_model']
        st.session_state['test_size'] = st.session_state['temp_test_size']
        st.session_state['random_state'] = st.session_state['rand_state']
        st.session_state['train'] = True

    # Getting the number of classes
    num_classes = len(st.session_state['df'][st.session_state['target']].unique())

    # Creating a form for selecting the linear model
    with st.form(key='linear_model_multi'):
        st.write('Number of Classes: ', num_classes)
        st.selectbox("Choose a Model for Classification", model_type, key='temp_linear_model')
        st.number_input("Choose the Test Data Size Ratio", key='temp_test_size',
                                                        value=0.2, step=0.05, min_value=0.05, max_value=0.9)
        st.number_input("Choose a Random State", key='rand_state',
                                                        value=42, step=1, min_value=0, max_value=1000)
        st.form_submit_button('Select', on_click=on_click)

# Checking if the data is uploaded
if st.session_state['df'] is not None:
    linear_model()
else:
    st.warning('Please Upload Your data via selecting the "Preprocess Data" option in the sidebar to proceed.', icon="⚠️")

# Training the selected model
if st.session_state['train']:
    model_type = st.session_state['linear_model']
    if model_type == 'Linear Support Vector Machine':
        model = LinearSupportVectorMachineModel()
        model.parameters()

    if model_type == 'Logistic Regression':
        model = LogisticRegressionModel()
        model.parameters()
    
    st.session_state['train'] = False








