# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('🚗 Traffic Volume Predictor') 
st.write("Utalize our advanced Machine Learning application to predict traffic volume.")

# Display an image of penguins
st.image('traffic_image.gif')

# Load the pre-trained model from the pickle file
model_pickle = open('traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the diamonds dataset
traffic_df = pd.read_csv('traffic_volume_clean.csv')

# Create a sidebar for input collection
st.sidebar.image('traffic_sidebar.jpg', caption= 'Traffic Volume Predictor')

st.sidebar.write('### Input Features')
st.sidebar.write('You can either upload your data file or mannually enter input features:')

# Option to upload a CSV file

with st.sidebar.expander('Option 1: Upload CSV file containing traffic details.', expanded=False):
    st.write('Upload a CSV file with diamond features:')
    df = st.file_uploader("", type=["csv"])
    st.write('Sample Data Format for Upload:')  
    sample_df = traffic_df.drop(columns = ['traffic_volume'])
    st.write(sample_df.head(5))
    st.warning('⚠️ Ensure your uploaded file has the same column names and data types as shown above.')

# Use values from uploaded file or fallback to fixed defaults if no file is uploaded

with st.sidebar.expander('Option 2: Fill out Form', expanded=False):
    st.write('Enter the traffic details manually using the form below:')
    with st.form(key='traffic_form'):
        holiday = st.selectbox('Choose whether today is a designated holiday or not', options = ['None','Columbus Day', 'Veterans Day', 'Thanksgiving Day',
       'Christmas Day', 'New Years Day', 'Washingtons Birthday','Memorial Day', 'Independence Day', 'State Fair', 'Labor Day','Martin Luther King Jr Day'])
        if holiday == 'None':
            holiday = None
        temp = st.number_input('Average temperaure in Kelvin', min_value =0.0, max_value=330.0, value = 290.0)
        rain = st.number_input('Amount of mm of rain that occurs in the hour', min_value =0.0, max_value=10000.0, value = 0.0)
        snow = st.number_input('Amount of mm of snow that occurs in the hour', min_value =0.0, max_value=1.0, value = 0.0)
        cloud = st.number_input('Percentage of cloud cover', min_value = 0.0, max_value=100.0, value = 50.0)
        weather = st.selectbox('Choose the current weather', options = ['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog',
       'Thunderstorm', 'Snow', 'Squall', 'Smoke'])
        month = st.selectbox('Choose day', options = ['January', 'February', 'March', 'April', 'May', 'June',
       'July', 'August', 'September', 'October', 'November', 'December'])
        day = st.selectbox('Choose day', options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        hour = st.selectbox('Choose hour', options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        submit_button = st.form_submit_button('Predict')

alpha = None
if submit_button:
    st.success("Form Data submitted successfully.")
elif df is not None:
    st.success("CSV File uploaded successfully.")
else:
    if alpha is None:
        st.info("Please upload a CSV file or fill out the form to predict traffic volume.")


# If a CSV is uploaded, read and use it for inputs instead of sliders
if df is None:
    # st.success("Form Data submitted successfully.")
    alpha = st.slider("Select the confidence level:", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Select the confidence level for the prediction intervals")
    alpha_v = alpha
    encode_df = traffic_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])

    # Combine the list of user data as a row to default_df
    
    encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, cloud, weather, month, day, hour]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha_v)
    pred_value = round(prediction[0],0)
    lower_limit = intervals[:, 0][0][0]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are non-zero
    lower_limit = max(0, lower_limit) # ensure lower limit is non-negative
    upper_limit = max(0, upper_limit) # ensure upper limit is non-negative

    st.write("## Predicting Traffic Volume...")
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.0f}")
    st.write(f"Using a **{(1-alpha_v)*100}% Confidence Interval**: [{lower_limit:.0f}, {upper_limit:.0f}]")
elif df is not None:
    # st.success("CSV File uploaded successfully.")
    input_df = pd.read_csv(df)
    alpha = st.slider("Select the confidence level:", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Select the confidence level for the prediction intervals")
    alpha_v = alpha

    for index, row in input_df.iterrows():
        holiday = row['holiday']
        temp = row['temp']
        rain = row['rain_1h']
        snow = row['snow_1h']
        cloud = row['clouds_all']
        weather = row['weather_main']
        month = row['month']
        day = row['weekday']
        hour = row['hour']

        # Encode the inputs for model prediction
        encode_df = traffic_df.copy()
        encode_df = encode_df.drop(columns=['traffic_volume'])

        # Combine the list of user data as a row to default_df
        encode_df.loc[len(encode_df)] = [holiday, temp, rain, snow, cloud, weather, month, day, hour]

        # Create dummies for encode_df
        encode_dummy_df = pd.get_dummies(encode_df)

        # Extract encoded user data
        user_encoded_df = encode_dummy_df.tail(1)

        # Get the prediction with its intervals
        prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha_v)
        pred_value = round(prediction[0],0)
        lower_limit = round(intervals[:, 0][0][0],0)
        upper_limit = round(intervals[:, 1][0][0],0)

        # Ensure limits are non-zero
        lower_limit = max(0, lower_limit) # ensure lower limit is non-negative
        upper_limit = max(0, upper_limit) # ensure upper limit is non-negative

        input_df.at[index, 'Predicted Traffic Volume'] = pred_value
        input_df.at[index, 'Lower Traffic Limit'] = lower_limit
        input_df.at[index, 'Upper Traffic Limit'] = upper_limit
  
    st.write(f"## Prediction Results with a {(1-alpha_v)*100}% Confidence Interval")
    st.write(input_df)

# Additional tabs for model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])

with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")

