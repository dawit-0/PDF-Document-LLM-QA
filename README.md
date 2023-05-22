# PDF Document LLM QA

## How it works

- Document is read using PyPdf
- Split the text into chunks with some overlapping
- Create embeddings for the chuncks using OpenAI embeddings
- Save the embeddings on disk for the file so we can load it next time instead of recreating
- Send QA query to openAI

## How to use

- Clone the repo
- Create a virtual environment and run `pip install -r requirements.txt`
- Run `streamlit run pdf_chat.py `
- This should automatically open the web app but if not you should see the link in your terminal
- Upload your desired document and ask away!

The web interface is built using [Streamlit](https://streamlit.io)
