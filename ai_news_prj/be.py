import os
import json
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

# JSON 디렉터리 경로 설정
raw_json_dir = "raw_ai_news_json"
summarized_json_dir = "sum_ai_news_json"

# JSON 파일에서 데이터를 읽어오는 함수
def load_json_files_to_dataframe(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # JSON 파일만 처리
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                # 필요한 키만 추출 (content, url, title)
                content = json_data.get("content", "")
                url = json_data.get("url", "URL 없음")
                title = json_data.get("title", "제목 없음")
                data.append({"content": content, "url": url, "title": title})
    # DataFrame 생성
    return pd.DataFrame(data)

# DataFrame에서 Document 객체로 변환하는 함수
def dataframe_to_documents(df):
    return [Document(page_content=row["content"], metadata={"url": row["url"], "title": row["title"]})
            for _, row in df.iterrows()]

# JSON 데이터를 DataFrame으로 로드
raw_articles_df = load_json_files_to_dataframe(raw_json_dir)
summarized_articles_df = load_json_files_to_dataframe(summarized_json_dir)

print("================= JSON 데이터를 테이블로 불러오기 완료 ====================")
print("\n[Raw Articles DataFrame]")
print(raw_articles_df.head())

print("\n[Summarized Articles DataFrame]")
print(summarized_articles_df.head())

# Retriever 생성 함수
def get_retriever(documents):
    # 텍스트 분할
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    splits = recursive_text_splitter.split_documents(documents)

    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    bm25_retriever = BM25Retriever.from_documents(documents)
    faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 가중치를 조합한 EnsembleRetriever 생성
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # 가중치 설정
    )
    return retriever

# DataFrame을 Document로 변환
raw_documents = dataframe_to_documents(raw_articles_df)
summarized_documents = dataframe_to_documents(summarized_articles_df)

# Retriever 생성
raw_retriever = get_retriever(raw_documents)
summerized_retriever = get_retriever(summarized_documents)

print("================= Retriever 생성 완료 ====================")

# 모델 초기화
model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)

# Prompt 템플릿
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the question using only the following context.
    You must include the article's URL and title in your answer whenever available.
    """),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

# DebugPassThrough 클래스
class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        # print("Debug Output:", output)
        return output

# 문서 리스트를 텍스트로 변환하는 단계
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):
        # 문서 제목, URL, 내용을 포맷팅
        context_text = "\n".join([
            f"Title: {doc.metadata.get('title', 'No Title')}\n"
            f"URL: {doc.metadata.get('url', 'No URL')}\n"
            f"Content:\n{doc.page_content}\n"
            for doc in inputs["context"]
        ])
        return {"context": context_text, "question": inputs["question"]}


# RAG 체인 생성
raw_rag_chain_debug = {
    "context": raw_retriever,            # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()       # 사용자 질문 확인
} | DebugPassThrough() | ContextToText() | contextual_prompt | model

summerized_rag_chain_debug = {
    "context": summerized_retriever,     # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()       # 사용자 질문 확인
} | DebugPassThrough() | ContextToText() | contextual_prompt | model

print("================= RAG 체인 생성 완료 ====================")

# while True:
#     query = input("질문을 입력하세요! 종료를 원한다면 exit을 입력하세요.")
#     if query == "exit":
#         break
#     print("question: " + query)
    
#     response = raw_rag_chain_debug.invoke(query)
#     print("RAG response : ", response)