import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer



if "df" not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None

norm_type = ['None', 'StandardScaler','MinMaxScaler','MaxAbsScaler',
                        'RobustScaler','QuantileTransformer']



def UploadData():
    form_hide = st.checkbox('Minimize', key= 'file_upload_form_hide')
    if not form_hide:
        with st.form(key = 'file_upload_form'):
            st.subheader('Preprocess and Gain Insights into Your Data')
            st.write("""Upload your data and preprocess it to gain insights into your data.""")
            uploaded_file = st.file_uploader('Upload your data here', type=['csv', 'xlsx']) 
            st.form_submit_button('Upload')
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('csv'):
                        st.session_state['df'] = pd.read_csv(uploaded_file)
                    else:
                        st.session_state['df'] = pd.read_excel(uploaded_file)
                    st.write('Data Uploaded Successfully')
                    st.write('***Data Shape***: ', st.session_state['df'].shape)
                    st.write(st.session_state['df'].head())
                    st.write('***Data Description***')
                    st.write(st.session_state['df'].describe())
                except Exception as e:
                    st.error('Error: {}'.format(e))
            else:
                st.write('Please upload a file')



def ProcessData(col_remove, target, norm_method):
    st.session_state['target'] = target
    temp_df = st.session_state['df']
    if len(col_remove) > 0:
        temp_df = temp_df.drop(col_remove, axis = 1)
    temp_df = temp_df.dropna()
    # temp_df = temp_df.reset_index(drop = True)
    if norm_method == "None":
        st.session_state['df'] = temp_df
        download_link(temp_df)
        return 
    
    target_df = temp_df[target]
    temp_df = temp_df.drop(target, axis = 1)
    scaler = None
    if norm_method == 'StandardScaler':
        scaler = StandardScaler()
    elif norm_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif norm_method == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif norm_method == 'RobustScaler':
        scaler = RobustScaler()
    elif norm_method == 'QuantileTransformer':
        scaler = QuantileTransformer()
    cols = temp_df.columns
    try:
        temp_df = scaler.fit_transform(temp_df)
    except Exception as e:
        st.error('Error: {}'.format(e))
        return
    temp_df = pd.DataFrame(temp_df, columns = cols)
    temp_df[target] = target_df
    st.session_state['df'] = temp_df  
    download_link(temp_df)


def download_link(temp_df):
    st.warning(""" If you need the copy of the cleaned data, please click the download button below.
               Otherwise, you can proceed to the next step by selecting the model you want to use.
               on the left side of the screen.""") 
    st.write('***Data Shape After Cleaning & Normalization***: ', temp_df.shape)
    st.write('***Overview of Data After Cleaning & Normalization***')
    st.write(temp_df.head())
    st.download_button('Download Data', data = temp_df.to_csv(), file_name = 'cleaned_data.csv', mime = 'text/csv')



def CleanAndNormalize():
    form_hide = st.checkbox('Minimize', key= 'normalize_data_form_hide')
    if not form_hide:
        temp_df = st.session_state['df']
        with st.form(key = 'normalize_data_form'):
           st.write('***Data Cleaning and Normalization***')
           st.warning('***Warning***: This will remove all rows with missing values')
           col_remove = st.multiselect('Select the columns to remove', temp_df.columns)
           target = st.selectbox('***Select the target variable***', options = temp_df.columns, index = None)
           norm_method = st.radio('***Select the normalization method***', norm_type)
           submitted = st.form_submit_button('Process Data')

        if submitted:
            if target:
                ProcessData(col_remove, target, norm_method)
            else:
                st.error('Please select a target variable')


UploadData()

if st.session_state['df'] is not None:
    CleanAndNormalize()




