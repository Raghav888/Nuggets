import streamlit as st
import Price
import portfolio

PAGES = {
    "Stock Price Prediction": Price,
    "Create Portfolio": portfolio
}
st.sidebar.title('Nuggets')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.price()
