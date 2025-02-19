import requests
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType


# DuckDuckGo search function using direct URL requests
def duckduckgo_search(query):
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text[
                :1000
            ]  # Return first 1000 characters of the HTML for simplicity
        else:
            return f"Error: Received status code {response.status_code}"
    except Exception as e:
        return f"Error: {e}"


# Streamlit app
def main():
    st.title("Local Web Search AI Agent")
    st.write(
        "This AI agent uses LangChain, Ollama, and DuckDuckGo to answer your questions."
    )

    # Input for user query
    user_query = st.text_input("Ask a question:")

    if st.button("Search"):
        if not user_query:
            st.error("Please enter a question.")
        else:
            # Define the DuckDuckGo tool
            def duckduckgo_tool(query):
                try:
                    results = duckduckgo_search(query)
                    return results
                except Exception as e:
                    return "Error: Could not fetch results from DuckDuckGo."

            # Define tools for the agent
            tools = [
                Tool(
                    name="DuckDuckGo Search",
                    func=duckduckgo_tool,
                    description="Useful for when you need to answer questions about current events or data. Input should be a search query.",
                )
            ]

            # Set up the LLM (Ollama)
            llm = Ollama(
                base_url="http://localhost:11434",
                model="deepseek-r1:1.5b",
            )

            # Define the prompt template
            prompt = PromptTemplate(
                input_variables=["input", "tools"],
                template="""
                You are a helpful AI assistant. Use the tools available to answer the following question.

                Question: {input}

                You have access to the following tools:
                {tools}

                To use a tool, respond in the following format:
                Action: {{tool_name}}
                Action Input: {{tool_input}}

                When you have the final answer, respond in the following format:
                Final Answer: {{answer}}

                If you cannot find the answer, respond with:
                Final Answer: I could not find the answer.
                """,
            )

            # Handle parsing errors with a custom function
            def handle_parsing_error(error):
                return f"Parsing error occurred: {str(error)}. Please try again."

            # Initialize an agent with tools and LLM
            try:
                agent_executor = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=handle_parsing_error,  # Custom error handling
                )

                # Run the agent with user input
                with st.spinner("Searching for answers..."):
                    response = agent_executor.run(user_query)
                    st.success("Answer:")
                    st.write(response)

            except Exception as e:
                st.error(f"An error occurred while running the agent: {e}")


# Run the Streamlit app
if __name__ == "__main__":
    main()
