import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os


# API Key

os.environ["OPENAI_API_KEY"] = "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"


def process_pdf(pdfdoc):

	pdfreader = PdfReader(pdfdoc.name)
	# get text
	raw_text = ''
	for i, page in enumerate(pdfreader.pages):
	    content = page.extract_text()
	    if content:
	        raw_text += content
	# Create tokens 

	text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
	)
	texts = text_splitter.split_text(raw_text)

	# get embeddings and save to vector DB

	embeddings = OpenAIEmbeddings()
	document_search = FAISS.from_texts(texts, embeddings)

	return document_search


def generate_quiz(document_search):

	chain = load_qa_chain(OpenAI(), chain_type="stuff")

	query = "Generate 5 MCQ quiz questions , also include the answers."
	docs = document_search.similarity_search(query)
	response = chain.run(input_documents=docs, question=query)
	st.write("Your AI crafted pop quiz is ready : ")
	st.image('/content/spongebob_dancing_1213.png')
	st.write(response)


def main():
    st.set_page_config("Quizify")
    st.header("Quizify ✎ : Craft quizzes from your own PDFs!")
    flag = False

    with st.sidebar:
        st.title("Upload your PDF here - ")
        pdfdoc = st.file_uploader("PDF upload")
        if st.button("Generate Quiz"):
            with st.spinner("Generating..."):
                flag = True
                #call_funcs()
                #st.success("Done")


    if flag == True:
        process_pdf(pdfdoc)
        generate_quiz()



if __name__ == "__main__":
    main()