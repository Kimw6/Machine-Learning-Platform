# Import necessary libraries
import streamlit as st
from Model_NN import create_and_train_nn

# Display subheader
st.subheader('Build Your Custom Neural Network for Classification')

# Define default session state variables
default_session_state = {
    'df': None,
    'target': None,
    'random_state': None,
    'train' : None,
}
# Initialize session state variables if not already present
for key, value in default_session_state.items():
    if key not in st.session_state:       
        st.session_state[key] = value

# Define types of classification
type_of_classification = ['Binary Classification', 'Multi-Class Classification']

# Define model selection function
def model_selection():

    # Define function to be executed on button click
    def on_click():
        if st.session_state['random_state'] is None:
            st.warning('Please select a random state.', icon="⚠️")
            return

        st.session_state['train'] = True

    # Get the number of classes in the target variable
    num_classes = len(st.session_state['df'][st.session_state['target']].unique())

    # Display information about the uploaded data
    with st.form(key='linear_model_multi'):
        st.write('Current Uploaded Data has `{}` classes for `{}`'.format(num_classes, st.session_state['target']))
        st.write('Current size of the data is `{}`'.format(st.session_state['df'].shape))
        st.write('Please select number hidden layers for your Neural Network')
        st.number_input("Select Number of Hidden Layers", min_value=1, max_value=5, step=1, key='num_layer', value=2)
        st.number_input("Select Random State", min_value=1, max_value=1000, step=1, key='random_state', value=42)
        st.number_input("Number of Epochs", min_value=1, max_value=100, step=1, key='epochs', value=10)
        st.form_submit_button('Select', on_click=on_click)

# Check if data is uploaded
if st.session_state['df'] is not None:
    model_selection()
else:
    st.warning('Please Upload Your data via selecting the "Preprocess Data" option in the sidebar to proceed.', icon="⚠️")

# Check if training is requested
if st.session_state['train']:
    num_layers = st.session_state['num_layer']
    random_state = st.session_state['random_state']
    df = st.session_state['df']
    target = st.session_state['target']
    
    # Create and train the neural network model
    model = create_and_train_nn(df, target, num_layers, random_state)
    
    # Display model summary
    model.summary(print_fn=lambda x: st.text(x))
    
    st.session_state['train'] = False










# default_session_state = {
#     'df': None,
#     'target': None,
#     'linear_model': None,
#     'test_size': None,
#     'random_state': None,
#     'train' : None,
# }
# for key, value in default_session_state.items():
#     if key not in st.session_state:       
#         st.session_state[key] = value


# model_type = ['Binary Classification', 'Multi-Class Classification']

# def model_selection():

#     def on_click():
#         if st.session_state['temp_linear_model'] is None:
#             st.warning('Please select a model.', icon="⚠️")
#             return
#         if st.session_state['temp_test_size'] is None:
#             st.warning('Please select a test size.', icon="⚠️")
#             return
#         if st.session_state['rand_state'] is None:
#             st.warning('Please select a random state.', icon="⚠️")
#             return
#         st.session_state['linear_model'] = st.session_state['temp_linear_model']
#         st.session_state['test_size'] = st.session_state['temp_test_size']
#         st.session_state['random_state'] = st.session_state['rand_state']
#         st.session_state['train'] = True

#     num_classes = len(st.session_state['df'][st.session_state['target']].unique())
#     with st.form(key = 'linear_model_multi'):
#         st.selectbox("Choose a Model for Classification", model_type, key= 'temp_linear_model')
#         st.form_submit_button('Select', on_click=on_click)


# if st.session_state['df'] is not None:
#     model_selection()
# else :
#     st.warning('Please Upload Your data via selecting the "Preprocess Data" option in the sidebar to proceed.', icon="⚠️")
# if st.session_state['train']:
#     model_type = st.session_state['linear_model']
#     if model_type == 'Decision Tree Classifier':
#         st.write('Imlement')

#     st.session_state['train'] = False
