from langchain_community.llms import Ollama  # Use langchain_community instead of langchain

# Load Llama 3.2 model from Ollama
llm = Ollama(model="llama3.2")

def ask_llm(question):
    """Function to ask Llama 3.2 a question and get a response."""
    response = llm.invoke(question)  # Use invoke() instead of direct call
    return response

if __name__ == "__main__":
    while True:
        query = input("Ask Llama 3.2: ")
        if query.lower() == "exit":
            break
        print("Response:", ask_llm(query))