import streamlit as st
from Model_NN import create_model, train_model
from utilityFunctions import handle_errors

st.subheader('`Build Your Custom Neural Network for Classification`')

default_session_state = {
    'df': None,
    'target': None,
    'train' : None,
}

for key, value in default_session_state.items():
    if key not in st.session_state:       
        st.session_state[key] = value

type_of_classification = ['Binary Classification', 'Multi-Class Classification']

def model_selection():
    def on_click():
        if st.session_state['random_state'] is None:
            st.warning('Please select a random state.', icon="⚠️")
            return

        st.session_state['train'] = True

    num_classes = len(st.session_state['df'][st.session_state['target']].unique())

    with st.form(key='linear_model_multi'):
        st.write('Current Uploaded Data has `{}` classes for `{}`'.format(num_classes, st.session_state['target']))
        st.write('Current size of the data is `{}`'.format(st.session_state['df'].shape))
        st.write('Please select number hidden layers for your Neural Network')
        st.number_input("Select Number of Hidden Layers", min_value=1, max_value=5, step=1, key='num_layer', value=2)
        st.number_input("Select Random State", min_value=1, max_value=1000, step=1, key='random_state', value=42)
        st.number_input("Number of Epochs", min_value=1, max_value=100, step=1, key='epochs', value=10)
        st.form_submit_button('Select', on_click=on_click)

if st.session_state['df'] is not None and st.session_state['target'] is not None:
    model_selection()
else:
    st.warning('Please Upload Your data via selecting the "Preprocess Data" option in the sidebar to proceed.', icon="⚠️")

if st.session_state['train']:
    num_layers = st.session_state['num_layer']
    random_state = st.session_state['random_state']
    df = st.session_state['df']
    target = st.session_state['target']
    try:
        model = create_model(df, target, num_layers, random_state)
        st.warning('Activation Functions: Relu for hidden layers')
        st.warning('Output Activation Function: Sigmoid for Binary Classification, Softmax for Multi-Class Classification')
        model.summary(print_fn=lambda x: st.text(x))
        train_model(model, st.session_state['epochs'], random_state)
    except Exception as e:
        handle_errors(e)
    st.session_state['train'] = False
