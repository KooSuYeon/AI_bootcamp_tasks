## 📋 Introduction
- 사용자의 AI 관련 질문을 관련 뉴스나 Youtube 영상 제공으로 정보의 확장을 도와주는 Chatbot

![alt text](image/running.gif)

---
## 📣 How To Use

```
1. Python version 확인
Python은 3.12 버전 기준

2. 필요한 패키지 설치
pip install -r requirements.txt

3. Docker 이미지를 pull하고 pull 한 이미지를 바탕으로 생성된 container를 실행
docker pull chromadb/chroma
docker run -p 9000:8000 chromadb/chrom

4. 로컬 구동
streamlit run fe.py

```

---
## 💻 Applied Technology
- Pandas 
- BeautifulSoup
- chromadb
- langchain
- pydantic
- streamlit

> API
- openAI
- youtubeAPI

---
## 🗝️ Key Summary

- [X] 뉴스 데이터 정적 크롤링
- [X] 요약된 뉴스 본문과 요약되지 않은 뉴스 본문 비교
- [X] Streamlit 이용한 UI 확인
- [X] 임베딩돤 뉴스 데이터 vetorDB 저장을 위한 Chroma Docker 도입
- [X] 사용자의 입력값에 따른 매체 선택 분류 기능 적용
- [X] Youtube API에 Agent를 도입해 관련 영상 가져오는 기능 도입
- [X] Streamlit 2차 연결

---
## 🎢 Timeline

### 1. AI News Crawling

> 2024/11/20
- 10,000 articles (2023/07/20 ~ 2024/11/20)
- including title, created_at, content -> json
- original site : https://www.aitimes.com/news/articleList.html


### 2. Create News Summarization
> 2024/11/21
- 100 articles (random)
- LLM : openAI chat-gpt-4o-mini
- embedding : text-embedding-ada-002

### 3. Create News Rag & Chatbot
> 2024/11/26
- Two rag chain (raw, summarized)
- BE : be.py
- FE : streamlit (fe.py)
- Find middle text hole

How To Use :
`streamlit run fe.py`

Test Screen : 

![alt text](image/image.png)


### 4. Create Youtube Agent
> 2024/12/10
- create vectorDB : embedding.ipynb
- agent : agent.ipynb

How To Use :
`Turn on the Chroma Docker & Go to agent.ipynb`

Test Screen :

![alt text](image/agent_1.png)

---

