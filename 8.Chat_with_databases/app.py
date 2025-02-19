import streamlit as st
import psycopg2
import ollama
import os
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_db():
    """Establish a connection to the PostgreSQL database."""
    try:
        dbname = os.getenv("DB_NAME")
        if not dbname:
            raise ValueError("DB_NAME environment variable is missing.")
        return psycopg2.connect(
            dbname=dbname,
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        return None
    except psycopg2.Error as e:
        st.error(f"Database connection failed: {e}")
        return None


def validate_sql(sql_query):
    # Remove AI-generated thought process (anything inside <think>...</think>)
    sql_query = re.sub(r"<think>.*?</think>", "", sql_query, flags=re.DOTALL).strip()
    
    # Use regex to extract a valid SQL query (basic SELECT, INSERT, UPDATE, DELETE)
    sql_pattern = r"(SELECT|INSERT|UPDATE|DELETE).*?;"
    sql_matches = re.findall(sql_pattern, sql_query, re.IGNORECASE | re.DOTALL)

    if sql_matches:
        # Extract and clean up the first valid SQL query
        sql_query = sql_query.strip()  # Clean up whitespace
        logger.info(f"Extracted SQL query: {sql_query}")  # Use logger instead of print for debugging
        return sql_query
    else:
        raise ValueError("Generated SQL is not a valid query.")


def generate_sql(query_text):
    """Generate SQL from a user query using Ollama DeepSeek."""
    prompt = f"""
    Convert this user question into a valid SQL query.
    IMPORTANT: Only return the SQL query itself, without any explanation, tags, or additional text.
    Do not include any text like '<think>' or any other non-SQL content.
    The response should be a valid SQL query that can be executed directly.

    User question: {query_text}

    SQL query:
    """

    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": prompt}],
        )
        sql_query = response["message"]["content"].strip()
        logger.info(f"Generated SQL: {sql_query}")
        sql_query = validate_sql(sql_query)
        return sql_query
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        st.error(f"Error generating SQL: {e}")
        return None


def execute_query(sql):
    """Execute an SQL query and return results."""
    conn = connect_db()
    if not conn:
        return None, "Failed to connect to the database."

    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        if cursor.description:  # Check if it's a SELECT query
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return columns, data
        else:
            conn.commit()
            return None, "Query executed successfully."
    except psycopg2.Error as e:
        return None, f"Database error: {e.pgerror}"
    except Exception as e:
        return None, f"Unexpected error: {e}"
    finally:
        cursor.close()
        conn.close()


def main():
    """Main Streamlit application function."""
    st.title("SQL Agent - Chat with Your Database")

    # User input for natural language query
    user_input = st.text_input("Ask your database a question:")

    if st.button("Generate SQL and Execute"):
        if not user_input.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Generating SQL query..."):
            sql_query = generate_sql(user_input)

        if sql_query:
            st.subheader("Generated SQL Query:")
            st.code(sql_query, language="sql")

            with st.spinner("Executing SQL query..."):
                columns, data = execute_query(sql_query)

            if columns:  # If it's a SELECT query with results
                st.subheader("Query Results:")
                st.dataframe([dict(zip(columns, row)) for row in data])
            else:  # For non-SELECT queries or errors
                st.write(data)

    if st.button("Clear"):
        st.experimental_rerun()


if __name__ == "__main__":
    main()
