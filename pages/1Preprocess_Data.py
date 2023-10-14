import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
      MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df


st.session_state.setdefault('df', None)
st.session_state.setdefault('target', None)
st.session_state.setdefault('col_remove', [])
st.session_state.setdefault('norm_method', None)

NORM_TYPE = [None, 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer']

def upload_file():
    st.write("""***Upload your data and preprocess it to gain insights into your data.***""")
    upload_file = st.file_uploader('Upload your data here', type=['csv', 'xlsx'])
    if upload_file is not None:
        if st.button('Upload'):
            try:
                temp_df = None
                if upload_file.name.endswith('csv'):
                    temp_df = pd.read_csv(upload_file)
                else:
                    temp_df = pd.read_excel(upload_file)
                st.write('Data Uploaded Successfully')
                st.write('***Data Shape***: ', temp_df.shape)
                st.write("***Overview of Data***")
                st.write(temp_df)
                st.write('***Data Description***')
                st.write(temp_df.describe())
                temp_df = temp_df.dropna()
                temp_df = temp_df.reset_index(drop=True)
                st.session_state['df'] = temp_df
            except Exception as e:
                st.error('Error: {}'.format(e))  
    st.write('***Or Use the Breast Cancer Dataset***') 
    if st.button('Load Breast Cancer Dataset'):
        try:
            temp_df = load_data()
            st.write('Data Uploaded Successfully')
            st.write('***Data Shape***: ', temp_df.shape)
            st.write("***Overview of Data***")
            st.write(temp_df)
            st.write('***Data Description***')
            st.write(temp_df.describe())
            temp_df = temp_df.dropna()
            temp_df = temp_df.reset_index(drop=True)
            st.session_state['df'] = temp_df
        except Exception as e:
            st.error('Error: {}'.format(e))   

def apply_normalization(normalization_type):
    normalization_functions = {
        None: None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer()
    }
    return normalization_functions[normalization_type]


def download_link():
    temp_df = st.session_state['df']
    st.write('***Data Shape After Cleaning & Normalization***: ', temp_df.shape)
    st.write('***Overview of Data After Cleaning & Normalization***')
    st.write(temp_df)
    st.write(temp_df.describe())

    st.warning(""" If you need a copy of the cleaned data, please click the download button below.
               Otherwise, you can proceed to the next step by selecting the model you want to use
               on the left side of the screen.""")
    
    st.download_button('Download Data', data=temp_df.to_csv(index=False), file_name='cleaned_data.csv', mime='text/csv')


def process_data():

    def on_click():
        if not st.session_state['temp_target']:
            st.error('Please select a target variable')
            return
        st.session_state['target'] = st.session_state['temp_target']

    st.warning('***Note***: This will remove all rows with missing values')
    cols = st.session_state['df'].columns
    with st.form(key = 'process_data'):
        st.multiselect('Select the columns to remove', cols, key='col_remove')
        st.selectbox(label ='***Select the target variable***', options=cols, 
                     index=None, key='temp_target')
        st.radio('***Select the normalization method***', NORM_TYPE, key='norm_method')
        form_sub = st.form_submit_button('Process Data', on_click=on_click)
  
def normalize_delete():
    temp_df = st.session_state['df']
    target = st.session_state['target']
    if len(st.session_state['col_remove']) > 0:
        temp_df = temp_df.drop(st.session_state['col_remove'], axis=1)

    norm_type = st.session_state['norm_method']
    norm_type = apply_normalization(norm_type)
    if norm_type is None:
        st.session_state['df'] = temp_df
        download_link()
        return
    else:
        target_df = temp_df[target]
        temp_df = temp_df.drop(target, axis=1)
        cols = temp_df.columns
        try:
            temp_df = norm_type.fit_transform(temp_df)
            temp_df = pd.DataFrame(temp_df, columns=cols)
            temp_df[target] = target_df
            st.session_state['df'] = temp_df
            download_link()
        except Exception as e:
            st.session_state['df'] = temp_df
            st.error('Error: {}'.format('Your data contains non-numeric values. Please remove them and try again.'))
            print(e)
            return  


    

upload_file()
if st.session_state['df'] is not None:
    process_data()
    if st.session_state['target'] is not None:
        normalize_delete()
