import streamlit as st
import pandas as pd
import re
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Data Analyst", layout="centered")
st.title("📊 AI Data Analyst (Text-to-SQL + Visualization)")
st.write("Ask questions about your database in plain English.")

# =========================
# DATABASE CONNECTION
# =========================
db = SQLDatabase.from_uri("sqlite:///student_grades.db")

# =========================
# LLM SETUP
# =========================
llm = ChatOllama(
    model="kimi-k2.5:cloud".strip(),
    temperature=0
)

# =========================
# SQL GENERATION PROMPT
# =========================
sql_prompt = ChatPromptTemplate.from_template("""
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

sql_chain = sql_prompt | llm | StrOutputParser()

# =========================
# ANALYSIS PROMPT
# =========================
analysis_prompt = ChatPromptTemplate.from_template("""
You are a data analyst.

Given the SQL query and its result, explain the answer clearly in simple English.

SQL Query:
{query}

Result:
{result}

Give a short, clear explanation.
""")

analysis_chain = analysis_prompt | llm | StrOutputParser()

# =========================
# GET SCHEMA
# =========================
schema = db.get_table_info()

# =========================
# USER INPUT
# =========================
question = st.text_input(
    "💬 Enter your question:",
    placeholder="e.g., Who scored highest in Math?"
)

# =========================
# MAIN EXECUTION
# =========================
if question:
    try:
        with st.spinner("🧠 Thinking..."):

            # =========================
            # Step 1: Generate SQL
            # =========================
            sql_query = sql_chain.invoke({
                "schema": schema,
                "question": question
            }).strip()

            # 🔥 CLEAN MARKDOWN
            sql_query = re.sub(r"```sql|```", "", sql_query).strip()

            # 🔥 EXTRACT ONLY SELECT QUERY
            match = re.search(r"(SELECT .*?;)", sql_query, re.IGNORECASE | re.DOTALL)
            if match:
                sql_query = match.group(1)

            st.subheader("🧠 Generated SQL")
            st.code(sql_query, language="sql")

            # =========================
            # 🔐 SAFETY CHECK
            # =========================
            if any(word in sql_query.lower() for word in ["drop", "delete", "truncate", "update"]):
                st.error("❌ Dangerous query blocked!")
                st.stop()

            # =========================
            # Step 2: Execute SQL
            # =========================
            result = db.run(sql_query)

            st.subheader("📈 Raw Result")
            st.write(result)

            # =========================
            # Convert to DataFrame
            # =========================
            try:
                df = pd.DataFrame(result)

                st.subheader("📊 Data Table")
                st.dataframe(df)

                # =========================
                # GRAPH SECTION
                # =========================
                st.subheader("📊 Visualization")

                if not df.empty:
                    numeric_cols = df.select_dtypes(include='number').columns

                    if len(numeric_cols) > 0:
                        chart_type = st.selectbox(
                            "Choose Chart Type",
                            ["Bar Chart", "Line Chart", "Area Chart"]
                        )

                        x_col = st.selectbox("X-axis", df.columns)
                        y_col = st.selectbox("Y-axis", numeric_cols)

                        chart_data = df.set_index(x_col)[y_col]

                        if chart_type == "Bar Chart":
                            st.bar_chart(chart_data)

                        elif chart_type == "Line Chart":
                            st.line_chart(chart_data)

                        elif chart_type == "Area Chart":
                            st.area_chart(chart_data)
                    else:
                        st.warning("No numeric columns available for visualization")

            except Exception:
                st.warning("Could not convert result to DataFrame")

            # =========================
            # Step 3: AI ANALYSIS
            # =========================
            analysis = analysis_chain.invoke({
                "query": sql_query,
                "result": str(result)
            })

            st.subheader("📊 AI Analysis")
            st.success(analysis)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Built with LangChain + Ollama + Streamlit + Pandas")