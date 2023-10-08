import streamlit as st
import linear as util
from sklearn.model_selection import train_test_split


default_session_state = {
    'df': None,
    'target': None,
    'model_type': None,
    'params': None,
    'split': None
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


app = util.LinearModelApp()
if app.upload_data():
    if app.select_target_and_split_data():
        app.render_models()





# if "df" not in st.session_state:
#     st.session_state.df = None
# if 'target' not in st.session_state:
#     st.session_state.target = None
# if 'model_type' not in st.session_state:
#     st.session_state.model_type = None
# if 'params' not in st.session_state:
#     st.session_state.params = None
# if 'split' not in st.session_state:
#     st.session_state.split = None




# st.subheader('Linear Models for Classification')

# st.write("""
# Linear models predict by calculating a weighted sum of input features. For classification, 
#         they estimate the likelihood of a data point belonging to a specific class. They aim 
#         to find a line (or in higher dimensions, a hyperplane) that best separates data points
#         based on their categories. "If the data involves more than two classes, we offer logistic regression. 
#          For binary classification, linear regression is also an option."
# """)



# if 'df' in st.session_state and st.session_state.df is not None and 'target' in st.session_state and st.session_state.target is not None:
#     st.warning("""Your preprocessing data is saved. You can proceed to train your model.
#                 If you want to change your data, please click on the button below""")
#     btn_change = st.button('Change Data')
#     if btn_change:
#         util.upload_data()
#     else:

#         sp_data = util.split_data()
#         if sp_data:
#             util.render_models()
            
# else: 
#     is_file_uploaded = util.upload_data()
#     if is_file_uploaded:
#         target_selected = util.select_target(st.session_state.df)
#         if target_selected:
#             sp_data = util.split_data()
#             if sp_data:
#                 util.render_models()
         
            
            

