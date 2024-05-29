from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
import glob

def get_pdf_text(pdf_folder):
    texts = ""
    pdf_files = glob.glob(f"{pdf_folder}/*.pdf")  # Find all PDF files in the folder
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts += page_text
    return texts

# Specify the folder containing the PDFs
pdf_folder = './dataset'

# Get the text from all PDFs in the folder
texts = get_pdf_text(pdf_folder)
# print(texts)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_text(texts)

# Embed
vectorstore = FAISS.from_texts(texts=splits, 
                                    embedding=OpenAIEmbeddings(api_key = 'sk-proj-SVFchhSzG9ysJqeNdH2lT3BlbkFJWpRZFgEiyq6hxi5vZqtF'))
vectorstore.save_local("faiss_index")