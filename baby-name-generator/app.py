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
    # Combine parts of father's and mother's names
    return f"{father_name[:len(father_name)//2]}{mother_name[len(mother_name)//2:]}"


def main():
    st.title("AI-Powered Baby Name Generator")

    # User inputs
    gender = st.radio("Select Gender", ["Male", "Female"], index=1)
    country = st.selectbox(
        "Select Country", ["USA", "Nepal", "India", "China", "Japan"]
    )
    starting_letter = st.text_input("Enter a starting letter (optional):", max_chars=1)
    father_name = st.text_input("Enter Father's Name (optional):")
    mother_name = st.text_input("Enter Mother's Name (optional):")
    birth_month = st.selectbox(
        "Select Birth Month",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    )
    birth_year = st.number_input(
        "Enter Birth Year", min_value=1900, max_value=2100, value=2023
    )

    # Generate prompt
    prompt = f"Generate 10 unique baby names for a {gender} child from {country}, born in {birth_month} {birth_year}."
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
            if "error" in names.lower():  # Check if an error message was returned
                st.error(names)  # Display the error message
            else:
                st.success("Suggested Names:")
                # Split the generated text into lines and display each name with meaning
                names_list = names.split("\n")
                for line in names_list:
                    if " - " in line:  # Check if the line contains a name and meaning
                        name, meaning = line.split(" - ", 1)
                        st.markdown(
                            f"**{name.strip()}** - *{meaning.strip()}*"
                        )  # Highlight name and meaning
                    else:
                        st.write(
                            line
                        )  # Fallback for lines without a clear name-meaning split


if __name__ == "__main__":
    main()
