from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain.schema import Document


load_dotenv() 
api_key = os.getenv("OPEN_API_KEY")

def get_retriever(context):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    vectorstore = FAISS.from_documents(documents=context, embedding=embeddings)

    return vectorstore.as_retriever()

# 파일을 읽어와서 text_list에 저장하는 함수
def load_files(file_path):
    if os.path.exists(file_path):  # 파일이 존재하는지 확인
        with open(file_path, 'r', encoding='utf-8') as file:  # 파일 열기
            content = file.read()  # 파일 내용 읽기
            return content
    else:
        print(f"File {file_path} not found.")

raw_articles = load_files("raw_text.txt")
summerized_articles = load_files("summerized_text.txt")

print("================= txt 불러오기 완료  ====================")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

def get_retriever(texts:str):

    # text_list를 Document 객체로 변환
    documents = [Document(page_content=texts)]

    recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )

    splits_recur = recursive_text_splitter.split_documents(documents)
    splits = splits_recur

    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    bm25_retriever = BM25Retriever.from_documents(documents)
    faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]  # 가중치 설정 (가중치의 합은 1.0)
            )

    return retriever

raw_retriever = get_retriever(raw_articles)
summerized_retriever = get_retriever(summerized_articles)


print("================= retriever 불러오기 완료  ===============")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 모델 초기화
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)

contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", """
     Answer the question using only the following context.
     """),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])


class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
    
    
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

# RAG 체인에서 각 단계마다 DebugPassThrough 추가
raw_rag_chain_debug = {
    "context": raw_retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model


# RAG 체인에서 각 단계마다 DebugPassThrough 추가
summerized_rag_chain_debug = {
    "context": summerized_retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model

print("================= rag_chain 불러오기 완료  ===============")