import sqlite3
import re
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent, SQLDatabaseToolkit

# Load the SQLite database
db = SQLDatabase.from_uri("sqlite:///titanic.db", include_tables=["titanic"])

# Load Llama 3.2 model from Ollama
llm = OllamaLLM(model="llama3")

# Create a SQL toolkit for the agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Define a single set of few-shot examples
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
    verbose=True,
    max_iterations=15,
    extra_prompt=(
    "You are an SQL expert. The database contains only one table: 'titanic'. "
    "Your task is to generate **only the SQL query** without executing it. "
    "DO NOT provide direct answers, explanations, or comments. "
    "ONLY output a single valid SQLite query, with no extra words.\n\n"
    "If the user asks something unrelated to SQL, respond with: 'Invalid question'.\n\n"
    "Examples:\n"
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
)

def extract_sql_query(response):
    """Extracts SQL query from LLM response."""
    sql_match = re.search(r"SELECT .*?;", response, re.IGNORECASE | re.DOTALL)
    
    if sql_match:
        return sql_match.group(0)  # Return the extracted SQL query

    # If response contains only a number, it means LLM skipped query generation
    if response.strip().replace(',', '').replace('.', '').isdigit():
        return None  # No SQL found

    return None  # No valid SQL found

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
            return f"❌ **Error:** Unexpected response format - {response_data}"

        print(f"\n[Raw LLM Response]: {raw_response}\n")  # Debugging step

        # Extract only the SQL query
        generated_sql = extract_sql_query(raw_response)

        if not generated_sql:
            return f"❌ **Error:** Could not extract a valid SQL query from response - {raw_response}"

        print(f"\n[Extracted SQL Query]: {generated_sql}\n")  # Debugging step

        # Execute the extracted query on SQLite
        conn = sqlite3.connect("titanic.db")
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        result = cursor.fetchall()
        conn.close()

        # Convert the result to a readable format
        if result:
            formatted_result = result[0][0] if len(result) == 1 and len(result[0]) == 1 else "\n".join([str(row) for row in result])
            return (
                f"✅ **SQL Query Used:**\n```sql\n{generated_sql}\n```\n\n"
                f"🔍 **Query Result:** `{formatted_result}`"
            )
        else:
            return (
                f"✅ **SQL Query Used:**\n```sql\n{generated_sql}\n```\n\n"
                f"⚠️ **No data found for the given query.**"
            )

    except sqlite3.OperationalError as e:
        return f"❌ **SQL Error:** {e}"
    except Exception as e:
        return f"❌ **Error:** {e}"
    
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a database question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        print("\nSQL Agent Response:", ask_sql_agent(user_input))