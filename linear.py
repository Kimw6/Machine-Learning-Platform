import streamlit as st
import pandas as pd

class LinearModelApp:
    def __init__(self, df=None, target=None, split=None, model_params=None):
        self.BINARY = [True, False]
        self.df = df
        self.target = target
        self.split = split
        self.model_params = model_params
        self.model_type = None

    def upload_data(self):
        st.warning("""**Upload Data to Train Your Model**
                    Ensure that your target variable is binary for binary classification,
                    or represents the appropriate classes in numerical form. Also, make sure 
                    that your data consists of numerical values only.""")
        with st.form(key='file_upload_form'):
            st.subheader('Upload Your Data for Analysis')
            uploaded_file = st.file_uploader('Upload your data here', type=['csv', 'xlsx'])
            btn_select = st.form_submit_button('Upload')
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(('csv', 'xlsx')):
                        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
                        self.df = df
                        st.session_state['df'] = df
                        st.success('File uploaded successfully.')
                        return True
                    else:
                        st.warning('Please upload a CSV or Excel file.')
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        return False
    # end upload_data


    def select_target_and_split_data(self):
        with st.form(key='target_and_split_form'):
            target = st.selectbox('Select Target Variable', self.df.columns)
            train_size = st.number_input('Train Size', value=0.8, step=0.1, min_value=0.1, max_value=0.9)
            test_size = st.number_input('Test Size', value=0.2, step=0.1, min_value=0.1, max_value=0.9)
            random_state = st.number_input('Random State', value=42, step=1, min_value=0, max_value=100)
            shuffle = st.selectbox('Shuffle', self.BINARY, index=0)
            btn_submit = st.form_submit_button('Select Target and Split Data')
            print(btn_submit)
            if btn_submit:
                if target:
                    if train_size + test_size == 1.0:
                        self.target = target
                        self.split = {'train_size': train_size, 'test_size': test_size, 
                                    'random_state': random_state, 'shuffle': shuffle}
                        st.session_state['target'] = target
                        return True
                    else:
                        st.warning('Train and test sizes should sum up to 1.0.')
                else:
                    st.warning('Please select a target variable.')
        
        return False
    # end select_target_and_split_data


    def train_linear_regression(self):
        st.write("""***Please adjust the parameters below to train your model***
                 more info on [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)""")
        with st.form(key='linear_regression_form'):
            col1, col2, col3 = st.columns(4)
            with col1:
                fit_intercept = st.selectbox('fit_intercept', self.BINARY, index=0)
            with col2:
                copy_X = st.selectbox('copy_X', self.BINARY, index=0)
            with col3:
                positive = st.selectbox('positive', self.BINARY, index=1)

            n_jobs = st.number_input('n_jobs', value=0, step=1, min_value=0, max_value=10)
            
            submitted = st.form_submit_button('Train Model')
            if submitted:
                self.model_params = {'fit_intercept': fit_intercept, 'copy_X': copy_X, 
                                     'n_jobs': n_jobs, 'positive': positive}
    # end train_linear_regression

    def train_logistic_regression(self):
        st.write("""***Please adjust the parameters below to train your model***
                 more info on [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)""")
        with st.form(key='logistic_regression_form'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                fit_intercept = st.selectbox('fit_intercept', self.BINARY, key='1', index=0)
                fit_intercept = st.selectbox('fit_intercept', self.BINARY, key='2', index=0)
            with col2:
                copy_X = st.selectbox('copy_X', self.BINARY, index=0)
            with col3:
                positive = st.selectbox('positive', self.BINARY, index=1)

            n_jobs = st.number_input('n_jobs', value=0, step=1, min_value=0, max_value=10)

            submitted = st.form_submit_button('Train Model')
            if submitted:
                self.model_params = {'fit_intercept': fit_intercept, 'copy_X': copy_X, 
                                     'n_jobs': n_jobs, 'positive': positive}
    # end train_logistic_regression
                
    def render_models(self):
        with st.form(key='model_selection_form'):
            st.subheader('Select One of the Linear Models Below')
            num_classes = self.df[self.target].nunique()
            st.warning(f"""Provided data has {num_classes} number of classes.""")

            if num_classes > 300:
                self.model_type = st.radio('Select Model', ['Logistic Regression'], key='logistic', label_visibility='hidden')
            else:
                self.model_type = st.radio('Select Model', ['Linear Regression', 'Logistic Regression'], key='linear', label_visibility='hidden')

            proceed = st.form_submit_button('Proceed')
            st.write(self.model_type)
            st.write(proceed)
        
            if proceed:
                if self.model_type == 'Linear Regression':
                    st.write("Linear Regression Model Training...")
                    # self.train_linear_regression()
                 
                elif self.model_type == 'Logistic Regression':
                    st.write("Logistic Regression Model Training...")
                    # self.train_logistic_regression()
                
            
            else:
                st.write("No model selected.")

                



    












# import streamlit as st
# import pandas as pd

# BINARY = [True, False]

# def upload_data():
#     st.warning("""**Upload Data to Train Your Model**
#                 Ensure that your target variable is binary for binary classification,
#                 or represents the appropriate classes in numerical form. Also, make sure 
#                 that your data consists of numerical values only.""")
#     with st.form(key='file_upload_form'):
#         st.subheader('Upload Your Data for Analysis')
#         uploaded_file = st.file_uploader('Upload your data here', type=['csv', 'xlsx'])
#         btn_select = st.form_submit_button('Upload')
#         if uploaded_file:
#             try:
#                 if uploaded_file.name.endswith(('csv', 'xlsx')):
#                     df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
#                     st.session_state['df'] = df
#                     st.success('File uploaded successfully.')
#                     return True
#                 else:
#                     st.warning('Please upload a CSV or Excel file.')
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#     return False    

# def select_target(df):
#     with st.form(key='select_target_form'):
#         target = st.selectbox('Select Target Variable', df.columns)
#         btn_select = st.form_submit_button('Select')
#         if btn_select:
#             if target:
#                 st.session_state['target'] = target
#                 return True
#             else:
#                 st.warning('Please select a target variable.')
#                 return False
#         else:
#             return False





        
# def split_data():
#     with st.form(key="slit_data_form"):
#         train_size = st.number_input('Train Size', value= 0.8, step=0.1, min_value=0.1, max_value=0.9)
#         test_size = st.number_input('Test Size', value= 0.2, step=0.1, min_value=0.1, max_value=0.9)
#         random_state = st.number_input('Random State', value= 42, step=1, min_value=0, max_value=100)
#         shuffle = st.selectbox('Shuffle', BINARY, index= 0)
#         btn_split = st.form_submit_button('Split Data')
#         if btn_split:
#             st.session_state['split'] = {'train_size': train_size, 'test_size': test_size, 
#                                             'random_state': random_state, 'shuffle': shuffle}
#             return True
#         else:
#             return False


# def LinearRegression():
#     st.write("""***Please adjust the parameters below to train your model***
#              more info on [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)""")
#     with st.form(key='linear_regression_form'):
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             fit_intercept = st.selectbox('fit_intercept', binary, index= 0)
#         with col2:
#             copy_X = st.selectbox('copy_X', binary, index= 0)
#         with col3:
#             positive = st.selectbox('positive', binary, index = 1)

#         n_jobs = st.number_input('n_jobs', value= 0, step=1, min_value=0, max_value=10)
        
#         submitted = st.form_submit_button('Train Model')
#         if submitted:
#             st.session_state['params'] = {'fit_intercept': fit_intercept, 'copy_X': copy_X, 
#                                           'n_jobs': n_jobs, 'positive': positive}
           



# def logistic_regression():
#     st.write("""***Please adjust the parameters below to train your model***
#              more info on [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)""")
#     with st.form(key='linear_regression_form'):
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             fit_intercept = st.selectbox('fit_intercept', BINARY, key = '1', index= 0)
#             fit_intercept = st.selectbox( 'fit_intercept', BINARY, key = '2',index= 0)
#         with col2:
#             copy_X = st.selectbox('copy_X', BINARY, index= 0)
#         with col3:
#             positive = st.selectbox('positive', BINARY, index = 1)

#         n_jobs = st.number_input('n_jobs', value= 0, step=1, min_value=0, max_value=10)

#         submitted = st.form_submit_button('Train Model')
#         if submitted:
#             st.session_state['params'] = {'fit_intercept': fit_intercept, 'copy_X': copy_X, 
#                                           'n_jobs': n_jobs, 'positive': positive}
#             pass

# def render_models():
#     st.subheader('Select One of the Linear Models Below')
#     st.warning("""Provided data has {} number of classes. """ 
#                 .format(st.session_state['df'][st.session_state['target']].nunique()))
#     model_type = None
#     # if st.session_state['df'][st.session_state['target']].nunique() > 2:
#     st.write('Makue sure to adjust the parameters for your model in render model function') 
#     if st.session_state['df'][st.session_state['target']].nunique() > 1000:
#         model_type = st.radio('Select Model', ['Linear Regression',], label_visibility='hidden')
#     else:
#         model_type = st.radio('Select a Model', ['Linear Regression', 'Logistic Regression'] , label_visibility='hidden')
    
#     proceed = st.button('Proceed')
#     if proceed:
#         st.session_state['model_type'] = model_type
#         if model_type == 'Linear Regression':
#             LinearRegression()
        
#         if model_type == 'Logistic Regression':
#             logistic_regression()
