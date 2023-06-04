# Convert any pdf to an audiobook

1. Built using meta's [MMS model](https://github.com/facebookresearch/fairseq/tree/main/examples/mms), streamlit and modal labs.
2. Each page is run in parallel, prone to CUDA out of memory errors (very fast though if you're lucky). 
3. Costs can add up pretty quickly so keep an eye out for it.

## Setup

1. Sign up for modal labs and get an API key
2. `poetry install`
3. `modal serve server.py`
4. Add `URL = "<your deployement url>` to .env file
5. `streamlit run client.py`

Enjoy!
