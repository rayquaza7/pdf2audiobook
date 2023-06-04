import os

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("URL")

st.title("Make any pdf an audiobook")

uploaded_file = st.file_uploader("Upload Files", type=["pdf"])


if uploaded_file is not None:
    if st.button("Generate"):
        # send file to URL
        with st.spinner("Generating audio..."):
            response = requests.post(URL, files={"file": uploaded_file})
            print(response.status_code)
            res = response.json()
            audio = np.array(res["audio"])
            st.audio(audio, sample_rate=res["sample_rate"])
