
import base64
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

st.title("Detecting Malicious URL")

st.write("The tabular column shows the type of the malicious URLs and the count of malicious URLs present in that dataset")

data = pd.read_csv('malicious_phish.csv')
count = data.type.value_counts()
x=count.index

plt.barplot(x=count.index, y=count)
plt.xlabel('Types')
plt.ylabel('Count');
st.pyplot(plt)

classifier_name = st.selectbox("Select a Machine Learning Technique",("Select","Decision Tree Classifier","Random Forest Classifier"))

accuracy = 90.96
accuracy1 = 91.50
a = "Decision Tree Classifier"
b = "Random Forest Classifier"

if classifier_name == a:
    accuracy = 90.96
    precision = 0.93
    recall = 0.97
    f1_score = 0.95
    
    st.write("Classification Report of Decision Tree Classifier")
    st.write("Accuracy  : ",accuracy)
    st.write("Precision : ",precision)
    st.write("Recall    : ",recall)
    st.write("F1-Score  : ",f1_score)

if classifier_name == b:
    accuracy1 = 91.50
    precision1 = 0.94
    recall1 = 0.98
    f1_score1 = 0.95
    
    st.write("Classification Report of Decision Tree Classifier")
    st.write("Accuracy  : ",accuracy1)
    st.write("Precision : ",precision1)
    st.write("Recall    : ",recall1)
    st.write("F1-Score  : ",f1_score1)
    
st.write("The Machine Learning Application Technique with more Accuracy is ",b,"with an accuracy of ",accuracy1)

    


    
