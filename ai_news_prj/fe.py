import os
from dotenv import load_dotenv
import streamlit as st
import be

# Streamlit 기본 설정
st.markdown(
    """
    <h1 style="text-align: center;">🤖 AI 정보 확장 챗봇 🤖</h1>
    """,
    unsafe_allow_html=True
)

st.markdown("--- ")

# Streamlit 기본 설정
st.markdown(
    """
    <h3>📈 Trend </h3>
    """,
    unsafe_allow_html=True
)
recent_docs = be.get_recent_docs()

# Adjust column widths by specifying the relative weights
col1, col2, col3= st.columns([2, 2, 2])  # 5 columns with equal width distribution
card_height = 300

# Loop through the recent_docs to populate each column with a card
for idx, doc in enumerate(recent_docs[:3]):  # Limit to the first 5 documents
    col_idx = idx % 5
    
    # Create containers in a loop for each document
    with eval(f'col{col_idx + 1}'):
        container = st.container(height=card_height)
        container.markdown(f"{doc['title']}")
        container.markdown(f"**{doc['other']}**")
        container.markdown(f"**Link:** {doc['url']}")

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("질문을 입력하세요! 종료하려면 'exit'을 입력하세요."):
    # 종료 조건 처리
    if user_input.lower() == "exit":
        st.info("채팅을 종료합니다.")
    else:
        # 사용자 메시지를 히스토리에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG 모델 호출
        with st.chat_message("assistant"):
            try:
                extract = be.extract_assistant(user_input)
                media = extract.get("action")
                keyword = extract.get("search_keyword")

                if media == "video":
                    st.subheader(f"🔍 {media} 매체로 원하시는 정보를 보여줄게요 ...")
                    st.session_state.messages.append({"role": "assistant", "content":f"🔍 {media} 매체로 원하시는 정보를 보여줄게요 ..."})
                    st.markdown(f"--- ")
                    results = be.search_youtube_videos(keyword)
                    assistant = be.print_videos_information(results)
                    st.markdown(assistant)
                    st.session_state.messages.append({"role": "assistant", "content":assistant})
                elif media == "news":
                    st.subheader(f"🔍 {media} 매체로 원하시는 정보를 보여줄게요 ...")
                    st.session_state.messages.append({"role": "assistant", "content":f"🔍 {media} 매체로 원하시는 정보를 보여줄게요 ..."})
                    st.markdown(f"--- ")
                    results = be.queryDB(keyword)
                    assistant = be.print_news_information(results)
                    st.markdown(assistant)
                    st.session_state.messages.append({"role": "assistant", "content":assistant})
                else:
                    error_message = "😅 본 서비스는 AI 관련 질의만 가능하세요... 🙏"
                    st.subheader(error_message)
                    st.session_state.messages.append({"role": "assistant", "content":error_message})

            except Exception as e:
                st.error(f"RAG 호출 중 오류가 발생했습니다: {str(e)}")
