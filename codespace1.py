import streamlit as st
import nltk
from rank_bm25 import BM25Okapi  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import re
from nltk.stem import WordNetLemmatizer 
from collections import Counter
import pandas as pd
import plotly.express as px
import random
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="trial", page_icon="🖥️", layout="wide")
st.sidebar.title("BM25 SEARCH ENGINE")
st.sidebar.write('''BM25 scores are relative, higher scores = more relevant documents.

Only PDFs with text content work; scanned PDFs may require OCR.

Each preview shows first 500 characters.

The web app is fully interactive and updates results in real-time.'''
)

uploaded = st.file_uploader("📂 Upload the files(txt or pdf)", accept_multiple_files=True, type=["txt", "pdf"])
query = st.text_input("🔍 Enter your search query:")

stop_words = set(stopwords.words('english'))

def readwtv(files):
    docs = []
    filenames = []
    for file in files:
        if file.type == "application/pdf":
            # pdfs
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)
        else:
            # txt
            text = file.read().decode(errors="ignore")
            docs.append(text)
        filenames.append(file.name)
    return docs, filenames

def pp(text):
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


if uploaded:
    raw_docs, doc_names = readwtv(uploaded)
    tokenizedbs = [pp(doc) for doc in raw_docs]

    bmval = BM25Okapi(tokenizedbs)
    hassmth=False
    if query:
        tokenized_query = pp(query)
        scores = bmval.get_scores(tokenized_query)

        ranked_results = sorted(
            list(zip(doc_names, raw_docs, scores)),
            key=lambda x: x[2],
            reverse=True
        ) #creatinga list of tuples, sorted via 3rd elememt =>x[2]=>scores=>bm25vales

        st.subheader("Search Results")
        for name, content, score in ranked_results:
            if score > 0:
                hassmth=True
                st.markdown(f"{name} — Score: `{score:.2f}`")
                st.markdown("---")
            else:
                st.info("Found Nothing Relevant in the given document")

        all_tokens = [token for doc in tokenizedbs for token in doc]
        allclean_tokens = [re.sub(r'[.,=\-]', '', t) for t in all_tokens if re.sub(r'[{}0123456789.,=\-]', '', t)]

        freq = Counter(allclean_tokens)
              # top 20 words
        n=20
        topn=freq.most_common(n)

        df_words = pd.DataFrame(topn, columns=["Word", "Frequency"])
        figure = px.bar(df_words, x="Word", y="Frequency", title=f"Top {n} Words in Documents")
        st.plotly_chart(figure, key=f"top_words_chart{random.randint(0,1000000)}")

