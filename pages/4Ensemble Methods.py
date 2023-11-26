import streamlit as st
from Model_Ensemble import RandomForestClassifierModel, GradientBoostingClassifierModel, AdaBoostClassifierModel

# Display subheader for the section
st.subheader("`Ensemble Methods for Classification`")

# Define default session state variables
default_session_state = {
    'df': None,
    'target': None,
    'linear_model': None,
    'test_size': None,
    'random_state': None,
    'train' : None,
}

# Initialize session state variables if they don't exist
for key, value in default_session_state.items():
    if key not in st.session_state:       
        st.session_state[key] = value

# Define the available model types
model_type = ['Random Forest Classifier', 'AdaBoost Classifier', 'Gradient Boosting Classifier']

def model_selection():
    # Function to handle button click event
    def on_click():
        # Check if all required fields are selected
        if st.session_state['temp_linear_model'] is None:
            st.warning('Please select a model.', icon="⚠️")
            return
        if st.session_state['temp_test_size'] is None:
            st.warning('Please select a test size.', icon="⚠️")
            return
        if st.session_state['rand_state'] is None:
            st.warning('Please select a random state.', icon="⚠️")
            return
        
        # Update session state variables with selected values
        st.session_state['linear_model'] = st.session_state['temp_linear_model']
        st.session_state['test_size'] = st.session_state['temp_test_size']
        st.session_state['random_state'] = st.session_state['rand_state']
        st.session_state['train'] = True

    # Get the number of unique classes in the target variable
    num_classes = len(st.session_state['df'][st.session_state['target']].unique())
    
    # Display the form for model selection
    with st.form(key='linear_model_multi'):
        st.write('Number of Classes: ', num_classes)
        st.selectbox("Choose a Model for Classification", model_type, key='temp_linear_model')
        st.number_input("Choose the Test Data Size Ratio", key='temp_test_size',
                                                        value=0.2, step=0.05, min_value=0.05, max_value=0.9)
        st.number_input("Choose a Random State", key='rand_state',
                                                        value=42, step=1, min_value=0, max_value=1000)
        st.form_submit_button('Select', on_click=on_click)

# Check if the data is uploaded
if st.session_state['df'] is not None:
    model_selection()
else:
    st.warning('Please Upload Your data via selecting the "Preprocess Data" option in the sidebar to proceed.', icon="⚠️")

# Check if the training flag is set
if st.session_state['train']:
    model_type = st.session_state['linear_model']
    
    # Create an instance of the selected model type
    if model_type == 'Random Forest Classifier':
        model = RandomForestClassifierModel()
        model.parameters()
    if model_type == 'Gradient Boosting Classifier':
        model = GradientBoostingClassifierModel()
        model.parameters()
    if model_type == 'AdaBoost Classifier':
        model = AdaBoostClassifierModel()
        model.parameters()

    # Reset the training flag
    st.session_state['train'] = False
