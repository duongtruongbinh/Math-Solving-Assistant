import streamlit as st
import google.generativeai as genai

# Cấu hình API key cho Google Generative AI
genai.configure(api_key='AIzaSyDCphh5DyMK8ZLCZKQM9MfhkmfY34CBKfI')

# Kiểm tra và thiết lập mô hình Gemini nếu chưa có trong session_state
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-pro"

st.title("Chatbox")

# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Hiển thị các tin nhắn trong lịch sử
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý nhập liệu từ người dùng
prompt = st.chat_input("Enter chatbot...")
if prompt:
    # Hiển thị tin nhắn của người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    # Thêm tin nhắn người dùng vào lịch sử chat
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Gọi API để tạo phản hồi
    response = genai.generate_content(
        model=st.session_state["gemini_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]],
        stream=True,
    )

    # Hiển thị phản hồi của trợ lý
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for part in response:
            full_response += part.choices[0].delta.get("content", " ")
        message_placeholder.markdown(full_response)
    # Thêm phản hồi trợ lý vào lịch sử chat
    st.session_state["messages"].append({"role": "assistant", "content": full_response})
