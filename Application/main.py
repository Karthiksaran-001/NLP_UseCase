"""
Created on Tue Jul  28 17:06:01 2019

@author: Karthik Saran
"""
from scipy.sparse import issparse
import streamlit as st
import pickle 
import streamlit as st
import nltk 

def Clean_text(data):
    '''This Method help to clean the data by removing the data stopwords punctuation'''
    nltk.download("stopwords")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    import re
    

    text = re.sub('[^a-zA-Z0-9]', ' ', data)
    text = text.lower()
    text = text.split() #
    text = [lemmatizer.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
 
    return [text]

def TFIDF(data):
    transform_model = "TFIDF.pkl"
    tfidf = pickle.load(open(transform_model , "rb"))
    vector = tfidf.transform(data).toarray()
    return vector 

def predction(message):
 
    import numpy as np
    nlp_model_file =  "nlp_model.pkl"
    
    
    nlp_model = pickle.load(open(nlp_model_file , "rb"))

    data = Clean_text(message)
    
    vector = TFIDF(data)
    result = nlp_model.predict(vector)
    accuracy = nlp_model.predict_proba(vector)
    ind = np.argmax(accuracy)
    accuracy = accuracy[0][ind] * 100
    return [result[0] , accuracy]


st.title("NLP Prediction WebApp")
st.markdown("Predict the Mail Subject")

with st.form('my_form'):
    global message
    message = st.text_area("Write Your Message")
    submitted = st.form_submit_button("Predict")

if(submitted):
    prediction_result = predction(message)[0]
    accuracy = predction(message)[1]

    st.write("Message Should be in  : \n" + prediction_result + " Class")
    st.write("Accuracy is : " +  str(accuracy) + "%")
