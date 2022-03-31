# To Run this Application - open Anaconda Prompt - "streamlit run StreamLitApp.py"
import pandas as pd
import numpy as np
import pickle
import streamlit as st 

pickle_in = open('classifier.pkl', 'rb')
classifer = pickle.load(pickle_in)


def Welcome():
    return "Welcome to my first Flask Application"

def main():
    st.title("Bank note Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">StreamLit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Variance", "type here")
    skewness = st.text_input("Skewness", "type here")
    curtosis = st.text_input("Curtosis", "type here")
    entropy = st.text_input("Entropy", "type here")
    result = ""
    if st.button("Predict"):
        result = classifer.predict([[variance,skewness,curtosis,entropy]])
        
    st.success("The Output is {}".format(result))
    if st.button("About"):
        st.text("Web App Using StreamLit")
        st.text("By Vijaynath")
    
if __name__== "__main__":
    main()
