# Math Solving Assistant

## 1. Overview

Our team has developed a chatbot website called **Math Solving Assistant** with the primary purpose of assisting users in solving mathematical problems. Additionally, the product supports users in chatting with the chatbot and analyzing images. The product is mainly built using Gemini's API, written in *Python*, and utilizes the *Streamlit* library to build the user interface. The product has been deployed on the domain [mathsolving-chatbot.streamlit.app](https://mathsolving-chatbot.streamlit.app/).

Here is a screenshot of the product interface:

![Homepage](https://i.imgur.com/m3dUJvr.png?1)

- On the left is the menu of the product, allowing users to switch between different functions.
  
- Below the menu is the image upload section for the integrated OCR feature.
  
- On the right is the main interface of the website, allowing users to interact, chat, and inquire with the chatbot.

## 2. Product Features

Our team's product includes three main models:

### **Chat with Gemini**:

The application will call Gemini's API to respond and chat with users. Additionally, we have integrated the OCR feature so users can input questions in image form, and the application will convert them into text for Gemini to respond to. Thus, users can solve problems such as chatting, questioning, translating, summarizing text, and more.

- Here is a screenshot of the chat with Gemini feature:

![Imgur](https://i.imgur.com/JFUUFX5.png)

- And here is a screenshot of the integrated OCR feature with an image input as a math problem:

![Imgur](https://i.imgur.com/OpheB7q.png)

- We can see that with the basic model of Gemini Pro 1.0, the chatbot feature works quite well. However, there are still some issues such as inaccurate answers and unstable model responses. Therefore, we will fine-tune the Gemini model on a math dataset to improve the quality of model responses. Details of this feature will be presented later.

### **Image Analysis**:

This is a convenient, supplementary feature to assist users. Based on Gemini Pro Vision's API, this feature allows users to upload images, and the product will describe the content of those images.

- Here is a screenshot of the image analysis feature:

![Imgur](https://i.imgur.com/vTUGMTY.png)

### **Math Problem Assistance**:

This is the main function of the product. As mentioned above, with the Gemini Pro 1.0 model, this feature already works quite well, but there are still some issues with the quality of model responses. Therefore, our team will fine-tune the Gemini model on a math dataset to improve the quality of model responses.

- The dataset to be used for fine-tuning is [MetaMathQA-40K](https://huggingface.co/datasets/meta-math/MetaMathQA-40K). This dataset consists of 40,000 math questions and corresponding answers from [MetaMath](https://huggingface.co/meta-math), collected by our team from Huggingface.

- The fine-tuning process will be conducted on Google Cloud Platform's Vertex AI.

    - Our team will convert the dataset into a jsonl file to upload to Vertex AI.
    
    - We will create a tuned-model on Vertex AI and perform fine-tuning.

    ![Imgur](https://i.imgur.com/9n0yb17.png)

    ![Imgur](https://i.imgur.com/cHGLaC4.png)

    - Here is the fine-tuning process on Vertex AI:

    ![Imgur](https://i.imgur.com/dmWIUU1.png)

- Here is a screenshot of the math problem assistance feature:

![Imgur](https://i.imgur.com/YsgCScz.png)

**Comparison of results between the old model and the fine-tuned model**

<img src="https://i.imgur.com/R7YQBUM.png" alt="Old Model" style="width: 45%; display: inline-block; margin-right: 5px;">
<img src="https://i.imgur.com/grzXGdz.png" alt="Fine-tuned Model" style="width: 45%; display: inline-block;">

- With the example above, we can see that the fine-tuned model provides more accurate results compared to the original model. Through multiple tests, our team has found that the fine-tuned model is also stable and provides accurate results for various questions.
