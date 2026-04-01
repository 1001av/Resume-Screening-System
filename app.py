import streamlit as st
from rag import ask

st.set_page_config(page_title="Resume Screening Chatbot")

st.title("💼 AI Resume Screener")

query = st.text_input("Enter job requirements:")

if st.button("Search"):
    if query:
        result = ask(query)
        st.write(result)