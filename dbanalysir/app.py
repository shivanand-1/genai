import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page Setup
st.set_page_config(page_title="Text-to-SQL App", layout="centered")
st.title("📊 Talk to Your Database")
st.write("Ask questions about the student grades database in plain English.")

# Database Connection
db = SQLDatabase.from_uri("sqlite:///student_grades.db")

# Local LLM (Ollama)
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# Prompt (LCEL Style)
prompt = ChatPromptTemplate.from_template("""
You are a senior data analyst and SQL expert.

Given the database schema below, write a correct SQL query
that answers the user's question.

Rules:
- Use only the tables and columns in the schema
- Do NOT explain anything
- Return ONLY the SQL query

Schema:
{schema}

Question:
{question}
""")

# LCEL Runnable Pipeline
sql_chain = (
    prompt
    | llm
    | StrOutputParser()
)

schema = db.get_table_info()