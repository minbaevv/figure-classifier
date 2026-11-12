import streamlit as st
import requests
from PIL import Image

st.title("Figure Predict")

api_url = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Images", width=200)

    if st.button("Pred"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        try:
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Предсказанная игра: {result['result']}")
            else:
                st.error(f"Error API: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к FastAPI серверу")

        except Exception as e:
            st.error(f"Произошла ошибак: {e}")