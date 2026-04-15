# 📊 AI Data Analyst (Text-to-SQL + Visualization)

An intelligent AI-powered data analyst that converts natural language questions into SQL queries, executes them on a database, visualizes the results, and provides human-like insights.

---

## 🚀 Features

* 🧠 Natural Language → SQL Query
* 🗄 Executes queries on SQLite database
* 📊 Automatic data visualization (Bar, Line, Area charts)
* 📈 Displays structured data tables
* 🤖 AI-generated insights (analysis of results)
* 🔐 Safe query execution (blocks harmful queries like DROP, DELETE)
* 🧹 Cleans LLM output (removes markdown errors)

---

## 🛠 Tech Stack

* **Frontend/UI**: Streamlit
* **LLM Framework**: LangChain
* **Model**: Ollama (Local LLM)
* **Database**: SQLite
* **Data Processing**: Pandas

---

## 📂 Project Structure

```
project/
│
├── app.py                 # Main application
├── student_grades.db     # SQLite database (ignored in Git)
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (optional)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
```

#### Activate:

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🧪 Example Queries

* Who scored highest in Math?
* Show average score per subject
* List top 5 students by score
* What is the minimum and maximum score in each subject?

---

## 📊 Example Output

* ✅ Generated SQL query
* ✅ Query result (table format)
* ✅ Graph visualization
* ✅ AI explanation of results

---

## 🔐 Security Features

* Prevents execution of:

  * DROP
  * DELETE
  * UPDATE
  * TRUNCATE
* Cleans markdown artifacts from LLM output

---

## 🧠 How It Works

1. User inputs a question
2. LangChain generates SQL using LLM
3. SQL is cleaned and validated
4. Query is executed on database
5. Results are visualized using Pandas + Streamlit
6. AI generates insights from results

---

## 🚀 Future Improvements

* 🎤 Voice input support
* 💬 Chat-based UI (like ChatGPT)
* 📥 Download results (CSV/Excel)
* 📊 Auto chart selection using AI
* 🌐 Deployment (Streamlit Cloud / Render)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 💼 Author

**Your Name**

* 💻 Aspiring Data Scientist / AI Engineer
* 🚀 Passionate about AI + Data + Automation

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
