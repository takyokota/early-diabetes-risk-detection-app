import streamlit as st
import pickle
import numpy as np


# Load data using pickle
def load_pickle_data():
    with open('App/model/svc_model.pkl', 'rb') as file:
        return pickle.load(file)


data = load_pickle_data()

svc = data['model']
df = data['dataframe']

# get the minimum and maximum age from dataframe for normalization on the 'Age'
min_age = df["Age"].min()
max_age = df["Age"].max()


# Questionnaire Form
st.title("Early Stage Diabetes Risk Prediction")

with st.form("app_form"):
    st.write("Answer all questions and press the 'Submit' button")

    age = st.number_input("Age:", min_value=0,
                          value=0, step=1, format='%d')
    st.write()

    gender = st.radio(
        "Gender:", ["Male", "Female"]
    )

    polyuria = st.radio(
        "Excessive Urination Amount:", ["Yes", "No"]
    )

    polydipsia = st.radio(
        "Excessive Thirst:", ["Yes", "No"]
    )

    swl = st.radio(
        "Sudden Weight Loss:", ["Yes", "No"]
    )

    weakness = st.radio(
        "Weakness:", ["Yes", "No"]
    )

    polyphagia = st.radio(
        "Excessive Hunger:", ["Yes", "No"]
    )

    gt = st.radio(
        "Genital Thrush (yeast infection):", ["Yes", "No"]
    )

    vb = st.radio(
        "Visual Blurring:", ["Yes", "No"]
    )

    itching = st.radio(
        "Itching:", ["Yes", "No"]
    )

    irritability = st.radio(
        "Irritability (mood swing):", ["Yes", "No"]
    )

    dh = st.radio(
        "Delayed Healing:", ["Yes", "No"]
    )

    pp = st.radio(
        "Partial Paresis (muscle weakness):", ["Yes", "No"]
    )

    ms = st.radio(
        "Muscle Stiffness:", ["Yes", "No"]
    )

    alopecia = st.radio(
        "Alopecia (hair loss):", ["Yes", "No"]
    )

    obesity = st.radio(
        "Obesity:", ["Yes", "No"]
    )

    submitted = st.form_submit_button("Submit")

    if submitted:
        # normalize age value
        age_normalized = (int(age) - min_age) / (max_age - min_age)

        yn_features = [age_normalized, gender, polyuria, polydipsia,
                       swl, weakness, polyphagia, gt, vb, itching,
                       irritability, dh, pp, ms, alopecia, obesity]

        # convert 'Yes' and 'No' into one and zero respectively
        for i in range(16):
            if yn_features[i] == 'Yes':
                yn_features[i] = 1
            elif yn_features[i] == 'No':
                yn_features[i] = 0
            elif yn_features[i] == 'Male':
                yn_features[i] = 1
            elif yn_features[i] == 'Female':
                yn_features[i] = 0

        input = np.array([yn_features])

        prediction = svc.predict(input)
        if prediction[0] == 1:
            st.write("**Prediction**: Positive")
        elif prediction[0] == 0:
            st.write("**Prediction**: Negative")


st.write("**Disclaimer**: The prediction is not intended to be a substitute for professional medical advice. \
                                 Always seek the advice of your doctor.")
