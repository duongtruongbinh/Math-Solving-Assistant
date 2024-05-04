import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

import pytesseract

import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
# working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

GOOGLE_API_KEY = 'AIzaSyChkUubQrrW-JPQs1adKRjusjvoPzeCSDA'
generation_config_gemini = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
}

safety_settings_gemini = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
system_instruction = "You are a professional math solving assistant for Vietnamese. Your answer must be in Vietnamese. Given a math problem, solve it step-by-step and provide a clear and concise explanation of the solution in Vietnamese."

# get response from Gemini-Pro-Vision model - image/text to text


def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result


working_dir = os.path.dirname(os.path.abspath(__file__))


def clear_history():
    # Clear chat history and messages
    if "history" in st.session_state:
        st.session_state.history = []
        st.session_state.messages = []

    # also clear the chat session at streamlit interface
    if "chat_session" in st.session_state:
        del st.session_state.chat_session


# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


def main():
    st.set_page_config(
        page_title="Gemini AI",
        page_icon="♊",
        layout="centered",
    )

    with st.sidebar:
        selected = option_menu('Menu AI',
                               ['Gemini ChatBot',
                                'Image Analysis',
                                'Math Solver'],
                               menu_icon='robot', icons=['chat-square-text-fill', 'badge-cc-fill', 'calculator-fill'],
                               default_index=0
                               )


# chatbot page
    if selected == 'Gemini ChatBot':
        upload_image = None

        # upload image
        upload_image = st.sidebar.file_uploader(
            'Chọn một hình ảnh chứa câu hỏi', type=["jpg", "png", "jpeg"])

        if selected != st.session_state.get('previous_model', None):
            clear_history()
            st.session_state['previous_model'] = selected
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-pro")

        # Initialize chat session in Streamlit if not already present
        if "chat_session" not in st.session_state:  # Renamed for clarity
            st.session_state.chat_session = model.start_chat(history=[])

        # Display the chatbot's title on the page
        st.title("🤖 Gemini ChatBot")

        # Display the chat history
        for message in st.session_state.chat_session.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

        # Input field for user's message
        user_prompt = st.chat_input("Đặt câu hỏi.....")
        if user_prompt:
            upload_image = None
            # Add user's message to chat and display it
            st.chat_message("user").markdown(user_prompt)

            # Send user's message to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(
                user_prompt)  # Renamed for clarity

            # Display Gemini-Pro's response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)

        if upload_image is not None:
            img = Image.open(upload_image)

            output_text = pytesseract.image_to_string(img, lang='vie+eng')
            st.chat_message("user").markdown(output_text)

            # Send user's message to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(
                output_text)

            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)
            # upload_image = None
    # Image captioning page
    elif selected == "Image Analysis":

        st.title("📷 Phân tích ảnh tự động")

        uploaded_image = st.file_uploader(
            "Tải ảnh lên...", type=["jpg", "jpeg", "png"])

        if st.button("Phân tích bức ảnh"):
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            default_prompt = "Mô tả bức ảnh bằng tiếng việt thật cụ thể, chi tiết"
            # get the caption of the image from the gemini-pro-vision LLM
            caption = gemini_pro_vision_response(default_prompt, image)

            with col2:
                st.info(caption)

    elif selected == "Math Solver":

        upload_image = None
        # upload image
        upload_image = st.sidebar.file_uploader(
            'Chọn một hình ảnh chứa câu hỏi', type=["jpg", "png", "jpeg"])

        if selected != st.session_state.get('previous_model', None):
            clear_history()
            st.session_state['previous_model'] = selected
        st.title("🧮 Chatbot hỗ trợ giải toán")

        vertexai.init(project="550674154992", location="us-central1")
        model = GenerativeModel(
            "projects/550674154992/locations/us-central1/endpoints/5172074109755981824",
            system_instruction=[system_instruction]
        )

        chat = model.start_chat(response_validation=False)
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = chat

        for message in st.session_state.chat_session.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

        user_prompt = st.chat_input("Đặt câu hỏi.....")
        if user_prompt:
            upload_image = None
            st.chat_message("user").markdown(user_prompt)
            gemini_response = st.session_state.chat_session.send_message(
                [user_prompt],
                generation_config=generation_config_gemini,
            )
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)
        if upload_image is not None:
            img = Image.open(upload_image)

            output_text = pytesseract.image_to_string(img, lang='vie+eng')
            user_prompt = output_text
            st.chat_message("user").markdown(user_prompt)

            # Send user's message to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(
                user_prompt)  # Renamed for clarity

            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)


if __name__ == "__main__":
    main()
