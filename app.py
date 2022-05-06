import streamlit as st
import pandas as pd

Color = st.get_option("theme.secondaryBackgroundColor")
s = f"""
<style>
div.stButton > button:first-child {{background-color: #fffff ; border: 2px solid {Color}; border-radius:5px 5px 5px 5px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

# caching.clear_cache()

activities = [
    "About this AI application",
    "Provide Information",
    "Data preprocessing",
    "Modeling",
    "Prediction",
]
st.sidebar.title("Navigation")
choices = st.sidebar.radio("", activities)

# ************************* Start About this AI application ***************************
if choices == "About this AI application":
    st.title("AI service for Student Assessment Performance")
    st.write(
        "This application creates initial prediction of student performance based on the trend of the school they attend"
    )

# ********************** Start Data upload and visualisation ***************************
if choices == "Provide Information":

    st.subheader("1. Data loading 🏋️")
    with st.form("stu_form"):
        st.write("Please Tell Us About Your Child")
        district = st.text_input(label="District")
        grade = st.number_input(label="Grade", format="%d")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("District", district, "Grade", type(grade))
