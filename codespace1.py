import streamlit as st
import nltk
from rank_bm25 import BM25Okapi  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="trial", page_icon="🖥️", layout="wide")
st.sidebar.title("BM25 SEARCH ENGINE")

uploaded = st.file_uploader("📂 Upload PDF/DOCX/TXT files", accept_multiple_files=True, type=["txt"])
query = st.text_input("🔍 Enter your search query:")

stop_words = set(stopwords.words('english'))