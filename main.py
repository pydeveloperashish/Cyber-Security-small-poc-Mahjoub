
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from htmlTemplates import css, bot_template, user_template

load_dotenv(find_dotenv())

llm = ChatOpenAI()

from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template(""" You are an expert in Cyber Security. Answer the following question based only on the provided context in detail:
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)


from langchain.chains import create_retrieval_chain


new_db = FAISS.load_local("faiss_index", 
                          embeddings = OpenAIEmbeddings(),
                           allow_dangerous_deserialization = True)


retriever = new_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def user_input(user_question):
    response = retrieval_chain.invoke({"input": str(user_question)})
    return response["answer"]




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    submit = st.button("Submit")
    if submit:
        response = user_input(user_question)
        with st.spinner("Processing"):
            st.write(response)

if __name__ == '__main__':
    main()