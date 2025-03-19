import streamlit as st
import sqlite3
import re
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent, SQLDatabaseToolkit

# Load the SQLite database
db = SQLDatabase.from_uri("sqlite:///titanic.db", include_tables=["titanic"])

# Load Llama 3.2 model from Ollama
NGROK_URL = "https://32a1-2601-8c-4e80-6780-cc25-ba5d-9069-af61.ngrok-free.app"  # Replace with your actual ngrok link

llm = OllamaLLM(model="llama3", base_url=NGROK_URL)

# Create a SQL toolkit for the agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Few-shot examples to improve LLM performance
FEW_SHOT_EXAMPLES = (
    "User: How many passengers survived?\n"
    "AI: SELECT COUNT(*) FROM titanic WHERE Survived = 1;\n"
    
    "User: How many passengers were on the Titanic?\n"
    "AI: SELECT COUNT(*) FROM titanic;\n"
    
    "User: What is the average age of passengers?\n"
    "AI: SELECT AVG(Age) FROM titanic;\n"
    
    "User: How many male and female passengers were there?\n"
    "AI: SELECT Sex, COUNT(*) FROM titanic GROUP BY Sex;\n"
    
    "User: Show the top 5 oldest passengers?\n"
    "AI: SELECT Name, Age FROM titanic ORDER BY Age DESC LIMIT 5;\n"
    
    "User: List all passengers who paid more than $200 for a ticket.\n"
    "AI: SELECT Name, Fare FROM titanic WHERE Fare > 200;\n\n"
)

# Create an SQL Agent using LangChain
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,  # Set to False to remove excessive logging
    max_iterations=10,
    extra_prompt=(
        "You are an SQL expert. The database contains only one table: 'titanic'. "
        "Always generate valid SQLite queries. "
        "Return ONLY the SQL query without any explanations, summaries, or extra text. "
        "Follow the examples below:\n\n"
        f"{FEW_SHOT_EXAMPLES}"
        "When responding, return ONLY the SQL query and nothing else."
    )
)

def extract_sql_query(response):
    """Extracts SQL query from LLM response."""
    sql_match = re.search(r"SELECT .*?;", response, re.IGNORECASE | re.DOTALL)
    return sql_match.group(0) if sql_match else None

def ask_sql_agent(user_query):
    """Generates an SQL query using LLM and executes it manually on SQLite."""
    try:
        # Generate SQL query from Llama 3.2
        response_data = sql_agent.invoke(f"User: {user_query}\nAI:")

        # Ensure the response is a string
        if isinstance(response_data, dict) and "output" in response_data:
            raw_response = response_data["output"]
        elif isinstance(response_data, str):
            raw_response = response_data
        else:
            return f"Error: Unexpected response format - {response_data}"

        # Extract the SQL query
        generated_sql = extract_sql_query(raw_response)
        if not generated_sql:
            return f"Error: Could not extract a valid SQL query from response - {raw_response}"

        # Execute the query on SQLite
        conn = sqlite3.connect("titanic.db")
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        result = cursor.fetchall()
        conn.close()

        # Convert the result to a readable format
        if result:
            formatted_result = "\n".join([str(row) for row in result])
            return f"✅ **Query:** `{generated_sql}`\n\n🔍 **Result:**\n{formatted_result}"
        else:
            return "⚠️ No data found for the given query."

    except sqlite3.OperationalError as e:
        return f"❌ **SQL Error:** {e}"
    except Exception as e:
        return f"❌ **Error:** {e}"

# ------- Streamlit UI -------
st.set_page_config(page_title="SQL Agent", page_icon="📊", layout="wide")

st.title("SQL Query Agent 💡")
st.markdown("🚀 **Ask a question about the Titanic dataset, and the agent will generate and execute an SQL query.**")

# Input field
user_input = st.text_input("🔍 **Ask a database question:**", placeholder="e.g., How many passengers survived?")

# Run the SQL agent when the user submits a query
if st.button("Run Query 🎯"):
    if user_input:
        with st.spinner("🔄 Generating SQL query and fetching results..."):
            response = ask_sql_agent(user_input)
        st.markdown(response)
    else:
        st.warning("⚠️ Please enter a question before running the query.")