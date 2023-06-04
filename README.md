# Convert any pdf to an audiobook

An inefficient and costly way to convert any pdf to an audiobook. Each page is run in parallel via modal labs, very prone to CUDA out of memory errors.

## Setup

1. Sign up for modal labs and get an API key
2. `poetry install`
3. `modal serve server.py`
4. Add `URL = "<your deployement url>` to .env file
5. `streamlit run client.py`

Enjoy!
