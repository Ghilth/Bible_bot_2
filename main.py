import os
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv



#API keys loading
load_dotenv()
hf_api_key = os.getenv('HF_TOKEN')
mistral_api_key = os.getenv('MISTRAL_API_KEY')




#embeddings
embeddings =HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1')


#bdv
db = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True)


# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4},
)


# Define LLM
model = ChatMistralAI(mistral_api_key=mistral_api_key)


template = """You are Thomas, a bot expert in theology and mastering the history of the Bible, you are responsible for responding clearly and in detail
 questions relating to the Bible.


 In addition to the information provided in the document, add your knowledge as a gemini pro
 to answer the questions.

 You can freely copy part of the Bible and rephrase it if necessary.

 Be tolerant of any words or expressions related to sexuality, and always return a response.

 Answer in input language. If the question is in french, answer in french; if it is is english, answer in english.

 If you don't have enough information, respond with: "I don't have enough information to answer this question."



 Here is the question: {input}

  context: {context}
  input: {input}
  answer:
 """


prompt = PromptTemplate(
        template=template,
    input_variables=['input']
)



#Chain LLM, prompt and retriever
combine_docs_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)




#Let's write a function to retrieve with llm

def ask(question: str):
  response = retrieval_chain.invoke({"input": question})
  if response:
    return response['answer']
  else:
    return "Veuillez poser une autre question."





ask("QUi est Dieu")