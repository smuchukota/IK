import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="CSV FAQ Agent", page_icon="📊", layout="centered")

st.title("📊 CSV FAQ Agent")
st.write("Upload CSV files and ask questions in plain English.")

# ---------------- API KEY ----------------
api_key = st.text_input("OpenAI API Key", type="password")

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

# ---------------- LOAD CSVs ----------------
dfs = {}

if uploaded_files:
    st.subheader("📄 File Preview")

    for file in uploaded_files:
        df = pd.read_csv(file)

        # normalize columns
        df.columns = df.columns.str.strip().str.lower()

        dfs[file.name] = df

        st.markdown(f"### {file.name}")
        st.dataframe(df.head())

# ---------------- LLM (optional for phrasing) ----------------
llm = None
if api_key:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# ---------------- CHAT STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- DISPLAY HISTORY ----------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
question = st.chat_input("Ask your question about the uploaded data...")

# ---------------- CORE ENGINE ----------------
def answer_question(question: str, dfs: dict):

    combined = pd.concat(dfs.values(), ignore_index=True)

    q = question.lower()

    # --------- 1. NUMERIC QUESTIONS ----------
    if any(word in q for word in ["total", "sum", "average", "avg", "mean"]):

        numeric_cols = combined.select_dtypes(include="number").columns

        if len(numeric_cols) == 0:
            return "I could not find numeric data in the uploaded files."

        col = numeric_cols[0]

        if "average" in q or "avg" in q or "mean" in q:
            return f"The average {col} is {round(combined[col].mean(), 2)}."

        if "total" in q or "sum" in q:
            return f"The total {col} is {round(combined[col].sum(), 2)}."

    # --------- 2. TEXT / POLICY QUESTIONS ----------
    text_cols = combined.select_dtypes(include="object").columns

    keywords = ["policy", "answer", "coverage", "hours", "limit", "rules", "details"]

    for col in text_cols:
        if any(k in col for k in keywords):
            value = combined[col].dropna()
            if not value.empty:
                return str(value.iloc[0])

    # --------- 3. FILTER-BASED SEARCH ----------
    # Try finding relevant rows using keyword match
    for col in combined.columns:
        if combined[col].dtype == "object":
            match = combined[combined[col].astype(str).str.contains(q, case=False, na=False)]
            if not match.empty:
                return match.iloc[0].to_string()

    # --------- 4. LLM FALLBACK (STRICT) ----------
    if llm:
        sample = combined.head(30).to_string()

        prompt = f"""
You are a strict data assistant.

Use ONLY the data below:

{sample}

Question:
{question}

Rules:
- Do NOT use external knowledge
- If answer is not present say:
  I could not find this information in the uploaded files
- Respond in clear customer-friendly English
"""

        return llm.invoke(prompt).content

    return "I could not find this information in the uploaded files."

# ---------------- RESPONSE HANDLING ----------------
if question and dfs:

    st.session_state.chat.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    answer = answer_question(question, dfs)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})