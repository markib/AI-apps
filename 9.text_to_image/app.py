import streamlit as st
from huggingface_hub import InferenceClient
import base64
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Initialize the Hugging Face Inference Client
client = InferenceClient(
    model="black-forest-labs/FLUX.1-dev", token=os.getenv("HUGGING_FACE_TOKEN")
)


def main():
    # Streamlit App Title
    st.title("Image Generation with FLUX.1-dev")

    # Input prompt from the user
    prompt = st.text_input("Enter a prompt to generate an image:", key="prompt_input")

    # Button to generate the image
    if st.button("Generate Image"):
        if prompt:
            # Generate the image using the Hugging Face model
            image = client.text_to_image(prompt)

            # Convert the image to a base64 string
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Display the generated image
            st.image(image, caption="Generated Image", use_container_width=True)

            # Optionally, provide a download link for the image
            st.markdown(
                f'<a href="data:image/png;base64,{img_str}" download="generated_image.png">Download Image</a>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Please enter a prompt to generate an image.")


# Entry point for the script
if __name__ == "__main__":
    main()
