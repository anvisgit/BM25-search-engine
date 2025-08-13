import streamlit as st
import nltk
from rank_bm25 import BM25Okapi  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="trial", page_icon="🖥️", layout="wide")
st.sidebar.title("BM25 SEARCH ENGINE")

uploaded = st.file_uploader("📂 Upload TXT files", accept_multiple_files=True, type=["txt"])
query = st.text_input("🔍 Enter your search query:")

stop_words = set(stopwords.words('english'))

def readwtv(files):
    docs = []
    filenames = []
    for file in files:
        content = file.read().decode(errors="ignore")
        docs.append(content)
        filenames.append(file.name)
    return docs, filenames

def pp(text):
    tokens = word_tokenize(text.lower())  
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]  
    return tokens

if uploaded:
    raw_docs, doc_names = readwtv(uploaded)
    tokenizedbs = [pp(doc) for doc in raw_docs]

    bmval = BM25Okapi(tokenizedbs)

    if query:
        tokenized_query = pp(query)
        scores = bmval.get_scores(tokenized_query)

        ranked_results = sorted(
            list(zip(doc_names, raw_docs, scores)),
            key=lambda x: x[2],
            reverse=True
        )

        st.subheader("📊 Search Results")
        for name, content, score in ranked_results:
            if score > 0:
                st.markdown(f"**📄 {name}** — Score: `{score:.2f}`")
                st.write(content[:500] + ("..." if len(content) > 500 else ""))  
                st.markdown("---")
