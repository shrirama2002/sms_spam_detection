#import nltk 

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps= PorterStemmer()


#function for transforming text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for _ in text:
        if _.isalnum():
            y.append(_)
            
    text = y[:]
    y.clear()
    
    for _ in text:
        if _ not in stopwords.words('english') and _ not in string.punctuation:
            y.append(_)
    
    text = y[:]
    y.clear()
    
    for _ in text:
        y.append(ps.stem(_))
        
    return " ".join(y)
    
#load the vectorizer and model using pickle
tk = pickle.load(open("vectorizer.pkl",'rb'))
model = pickle.load(open("model.pkl",'rb'))

st.title(''':blue-background[SMS Spam Detection Model]''')
st.write("*:orange[Made by Shrirama Kamath]*")
st.write(":blue[During internship with Edunet ]")

#get user data
inputSMS = st.text_input("Enter the SMS here")

if st.button('Predict'):
    # 1. preprocess
    transformedSMS = transform_text(inputSMS)
    # 2. vectorize the input
    vectorizedSMS = tk.transform([transformedSMS])
    # 3. predict
    result = model.predict(vectorizedSMS)[0]
    # 4. Display
    if result == 1:
        st.header(''':red:red-background[SPAM]''')
    else:
        st.header(''':green[NOT SPAM]''') 