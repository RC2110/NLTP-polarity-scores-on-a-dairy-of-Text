import streamlit as st
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer

import glob

files = glob.glob("dairy/*txt")

sentiment = SentimentIntensityAnalyzer()

positivity=[]
negativity=[]

dates = [i.strip(".txt").strip("dairy/") for i in files]
for i in files:
    with open(i, 'r')as file:
        content= file.read()
        scores = sentiment.polarity_scores(content)
        positivity.append(scores['pos'])
        negativity.append(scores['neg'])


st.title("Dairy Tone")
st.header("Positivity")
figure1 = px.line(x=dates, y=positivity, labels={"x":"Dates", "y":"Positivity score"})
st.plotly_chart(figure1)

st.header("Negativity")
figure2= px.line(x=dates, y=negativity, labels={"x":"Dates", "y":"Negativity score"})
st.plotly_chart(figure2)



