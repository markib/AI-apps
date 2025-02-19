import streamlit as st
import ollama


def generate_names(prompt):
    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b", messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except ollama.OllamaError as e:
        return f"An error occurred: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def combine_names(father_name, mother_name):
    return f"{father_name[:len(father_name)//2]}{mother_name[len(mother_name)//2:]}"


def main():
    st.title("AI-Powered Baby Name Generator")

    # User inputs
    gender = st.radio("Select Gender", ["Male", "Female"], index=1)
    country = st.selectbox("Select Country", ["USA","Nepal", "India", "China", "Japan"])
    starting_letter = st.text_input("Enter a starting letter (optional):", max_chars=1)
    father_name = st.text_input("Enter Father's Name (optional):")
    mother_name = st.text_input("Enter Mother's Name (optional):")

    # Generate prompt
    prompt = f"Generate 10 unique baby names for a {gender} child from {country}."
    if starting_letter:
        prompt += f" The names should start with the letter {starting_letter.upper()}."
    if father_name and mother_name:
        combined_name = combine_names(father_name, mother_name)
        prompt += f" Include a name inspired by combining the parents' names: {combined_name}."

    # Generate names
    if st.button("Generate Names"):
        if not gender or not country:
            st.warning("Please select both gender and country.")
        else:
            names = generate_names(prompt)
            st.success("Suggested Names:")
            st.write(names)


if __name__ == "__main__":
    main()
