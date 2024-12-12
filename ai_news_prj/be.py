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

from pydantic import BaseModel, Field  # 데이터 검증과 직렬화를 위한 Pydantic 라이브러리
from typing import Literal  # 다양한 타입 힌팅 클래스들
from langchain_core.output_parsers import JsonOutputParser  # LLM의 출력을 JSON 형식으로 파싱하는 도구
from langchain_core.prompts import PromptTemplate 
from googleapiclient.discovery import build
from datetime import datetime
from langchain.vectorstores import Chroma
import chromadb
from typing import List
import pytz

load_dotenv() 
open_ai_key = os.getenv("OPEN_AI_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

def get_llm(api_key:str):
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
    return model

class AgentAction(BaseModel):
    """
    에이전트의 행동을 정의하는 Pydantic 모델
    Pydantic은 데이터 검증 및 관리를 위한 라이브러리입니다.
    """
    # Literal을 사용하여 action 필드가 가질 수 있는 값을 제한합니다
    action: Literal["video", "news", "not_supported"] = Field(
        description="에이전트가 수행할 행동의 타입을 지정합니다",
    )
    
    action_input: str = Field(
        description="사용자가 입력한 원본 질의 텍스트입니다",
        min_length=1,  # 최소 1글자 이상이어야 함
    )
    
    search_keyword: str = Field(
        description="""검색에 사용할 최적화된 키워드입니다.
        AI 관련 키워드일 경우 핵심 검색어를 포함하고,
        not_supported 액션의 경우 빈 문자열('')을 사용합니다""",
        examples=["ChatGPT tutorial", "머신러닝 입문 강의"]  # 예시 제공
    )

output_parser = JsonOutputParser(pydantic_object=AgentAction)
# print("출력 포맷 가이드 :", output_parser.get_format_instructions())


prompt = PromptTemplate(
            input_variables=["input"],  # 템플릿에서 사용할 변수들
            partial_variables={"format_instructions": output_parser.get_format_instructions()},
            template="""당신은 AI 관련 YouTube 영상을 검색하는 도우미입니다.
입력된 질의가 AI 관련 내용인지 먼저 확인하세요.

AI 관련 주제 판단 기준:
- AI 기술 및 정보 (머신러닝, 딥러닝, 자연어처리 등)
- AI 도구 및 서비스 (ChatGPT, DALL-E, Stable Diffusion 등)
- AI 교육 및 학습
- AI 정책 및 동향

AI 관련 질의가 아닌 경우:
- action을 "not_supported"로 설정
- search_keyword는 빈 문자열로 설정

AI 관련 질의인 경우:
1. action을 "news" 또는 "video" 중에서 선택하세요.

- "video":
    - "영상", "비디오", "동영상"이라는 단어가 포함된 경우
- "news":
    - 분석, 배경 지식이 중요하거나 "뉴스"라는 단어가 포함된 경우.
    
    예제:
    - "비전 프로에 관련된 영상을 찾아줘" → "video"
    - "비전 프로의 최근 소식을 알려줘" → "news"
    - "긴급한 비전 프로 발표 영상을 보여줘" → "video"
    - "비전 프로에 대한 분석 기사를 찾아줘" → "news"
    - "gpt 관련 영상을 추천해줘" → "video"
     
    단어 분석:
        1. "영상", "비디오", "동영상" 또는 이와 유사한 단어가 포함되면 항상 "video"를 선택합니다.
        2. 위 단어가 없으면 "news"를 선택합니다.
    
2. 검색 키워드 최적화:
   - 핵심 주제어 추출
   - 불필요한 단어 제거 (동영상, 찾아줘 등)
   - 전문 용어는 그대로 유지

분석할 질의: {input}
{format_instructions}""")


def extract_assistant(order:str):
    llm = get_llm(open_ai_key)

    extract_chain = prompt | llm | output_parser
    extract = extract_chain.invoke({"input": order})

    # print(f"🧚 {extract.get("action")} 매체로 원하시는 정보를 보여줄게요!")
    # print("===============================================")

    return extract

# 유튜브 API 호출 함수
def search_youtube_videos(query: str) -> str:
    api_key = youtube_api_key  # 유튜브 API 키 입력
    youtube = build("youtube", "v3", developerKey=api_key)

    # 유튜브에서 검색
    search_response = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=5,  # 최대 검색 결과 수
        type="video",  # 비디오만 검색
        order="viewCount"  # 조회수 순으로 정렬
    ).execute()

    # 검색 결과 정리
    results = []
    for item in search_response.get("items", []):
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        video_id = item["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # 비디오 정보 조회
        video_response = youtube.videos().list(
            id=video_id,
            part="snippet,statistics"
        ).execute()

        snippet = video_response["items"][0]["snippet"]
        statistics = video_response["items"][0]["statistics"]
        likes_count = statistics.get("likeCount", 0)
        view_count = statistics.get("viewCount", 0)

        short_description = (description[:150] + '...') if len(description) > 150 else description

        utc_time = snippet["publishedAt"]
        utc_time_dt = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
        kst_tz = pytz.timezone('Asia/Seoul')
        kst_time = utc_time_dt.astimezone(kst_tz).strftime('%Y-%m-%d %H:%M:%S')


        # 결과 리스트에 추가
        results.append({
            "title": title,
            "url": video_url,
            "description": short_description,
            "likes": likes_count,
            "views": view_count,
            "upload_time": kst_time
        })

    return results

def queryDB(query):
    client = chromadb.HttpClient(host="localhost", port=9000)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=open_ai_key)
    db = Chroma(client=client, collection_name = "articles1", embedding_function = embeddings)
    docs = db.similarity_search_with_score(query)
    return docs

def get_recent_docs():
    client = chromadb.HttpClient(host="localhost", port=9000)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=open_ai_key)
    db = Chroma(client=client, collection_name="articles_recent", embedding_function=embeddings)
    
    document_list = []
    # Retrieve all documents from the collection
    all_docs = db.get()

    # Iterate through the documents and extract relevant fields
    for i in range(5):
        title = all_docs.get("ids")[i]
        other = all_docs.get("metadatas")[i].get('sub')
        url = all_docs.get("metadatas")[i].get('url')

        # Create a dictionary for each document
        document_dict = {
            'title': title,
            'other': other,
            'url': url
        }

        # Add the dictionary to the list
        document_list.append(document_dict)

    return document_list


def print_videos_information(videos: List):
    answer = ""
    cnt = 0
    for result in videos:
        title = result['title']
        video_url = result['url']
        description = result['description']
        likes = result['likes']
        views = result['views']
        upload_time = result["upload_time"]
        cnt += 1

        answer += f"**{cnt}.**\n"  # 숫자 강조
        answer += f"- **Title:** {title}\n"  # 굵게 표시
        answer += f"- **URL:** [{video_url}]({video_url})\n"  # URL에 링크 추가
        answer += f"- **Description:** {description}\n"
        answer += f"- **Likes:** {likes}\n"
        answer += f"- **Views:** {views}\n"
        answer += f"- **Upload Time:** {upload_time}\n"
        answer += "---\n"  # 구분선 추가

    return answer




def print_news_information(results: List):
    # 정렬: 결과를 (score, Document) 튜플로 변환 후 score에 따라 내림차순으로 정렬
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    answer = ""
    cnt = 0
    for result in sorted_results:
        document = result[0]
        metadata = document.metadata
        title = metadata.get('title', 'N/A')
        url = metadata.get('url', 'N/A')
        sub = metadata.get('sub', 'N/A')
        page_content = document.page_content
        score = result[1]
        cnt += 1

        answer += f"**{cnt}.**\n"  # 항목 번호 강조
        answer += f"- **Title:** {title}\n"
        answer += f"- **URL:** [{url}]({url})\n"  # URL을 링크로 표시
        answer += f"- **Content:** {page_content}\n"
        answer += f"- **Other:** {sub}\n"
        answer += f"- **Relevance:** {score}\n"
        answer += "---\n"  # 구분선 추가

    return answer




"""
이전 retriever로 갖고 왔던 방식 
"""



# # .env 파일에서 API 키 로드
# load_dotenv()
# api_key = os.getenv("OPEN_API_KEY")

# # JSON 디렉터리 경로 설정
# raw_json_dir = "raw_ai_news_json"
# summarized_json_dir = "sum_ai_news_json"

# # JSON 파일에서 데이터를 읽어오는 함수
# def load_json_files_to_dataframe(directory):
#     data = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".json"):  # JSON 파일만 처리
#             file_path = os.path.join(directory, filename)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 json_data = json.load(file)
#                 # 필요한 키만 추출 (content, url, title)
#                 content = json_data.get("content", "")
#                 url = json_data.get("url", "URL 없음")
#                 title = json_data.get("title", "제목 없음")
#                 data.append({"content": content, "url": url, "title": title})
#     # DataFrame 생성
#     return pd.DataFrame(data)

# # DataFrame에서 Document 객체로 변환하는 함수
# def dataframe_to_documents(df):
#     return [Document(page_content=row["content"], metadata={"url": row["url"], "title": row["title"]})
#             for _, row in df.iterrows()]

# # JSON 데이터를 DataFrame으로 로드
# raw_articles_df = load_json_files_to_dataframe(raw_json_dir)
# summarized_articles_df = load_json_files_to_dataframe(summarized_json_dir)

# print("================= JSON 데이터를 테이블로 불러오기 완료 ====================")
# print("\n[Raw Articles DataFrame]")
# print(raw_articles_df.head())

# print("\n[Summarized Articles DataFrame]")
# print(summarized_articles_df.head())

# # Retriever 생성 함수
# def get_retriever(documents):
#     # 텍스트 분할
#     recursive_text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=20,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     splits = recursive_text_splitter.split_documents(documents)

#     # OpenAI 임베딩 모델 초기화
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
#     vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#     bm25_retriever = BM25Retriever.from_documents(documents)
#     faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#     # 가중치를 조합한 EnsembleRetriever 생성
#     retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, faiss_retriever],
#         weights=[0.5, 0.5]  # 가중치 설정
#     )
#     return retriever

# # DataFrame을 Document로 변환
# raw_documents = dataframe_to_documents(raw_articles_df)
# summarized_documents = dataframe_to_documents(summarized_articles_df)

# # Retriever 생성
# raw_retriever = get_retriever(raw_documents)
# summerized_retriever = get_retriever(summarized_documents)

# print("================= Retriever 생성 완료 ====================")

# # 모델 초기화
# model = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)

# # Prompt 템플릿
# contextual_prompt = ChatPromptTemplate.from_messages([
#     ("system", """
#     Answer the question using only the following context.
#     You must include the article's URL and title in your answer whenever available.
#     """),
#     ("user", "Context: {context}\\n\\nQuestion: {question}")
# ])

# # DebugPassThrough 클래스
# class DebugPassThrough(RunnablePassthrough):
#     def invoke(self, *args, **kwargs):
#         output = super().invoke(*args, **kwargs)
#         # print("Debug Output:", output)
#         return output

# # 문서 리스트를 텍스트로 변환하는 단계
# class ContextToText(RunnablePassthrough):
#     def invoke(self, inputs, config=None, **kwargs):
#         # 문서 제목, URL, 내용을 포맷팅
#         context_text = "\n".join([
#             f"Title: {doc.metadata.get('title', 'No Title')}\n"
#             f"URL: {doc.metadata.get('url', 'No URL')}\n"
#             f"Content:\n{doc.page_content}\n"
#             for doc in inputs["context"]
#         ])
#         return {"context": context_text, "question": inputs["question"]}


# # RAG 체인 생성
# raw_rag_chain_debug = {
#     "context": raw_retriever,            # 컨텍스트를 가져오는 retriever
#     "question": DebugPassThrough()       # 사용자 질문 확인
# } | DebugPassThrough() | ContextToText() | contextual_prompt | model

# summerized_rag_chain_debug = {
#     "context": summerized_retriever,     # 컨텍스트를 가져오는 retriever
#     "question": DebugPassThrough()       # 사용자 질문 확인
# } | DebugPassThrough() | ContextToText() | contextual_prompt | model

# print("================= RAG 체인 생성 완료 ====================")

# # while True:
# #     query = input("질문을 입력하세요! 종료를 원한다면 exit을 입력하세요.")
# #     if query == "exit":
# #         break
# #     print("question: " + query)
    
# #     response = raw_rag_chain_debug.invoke(query)
# #     print("RAG response : ", response)