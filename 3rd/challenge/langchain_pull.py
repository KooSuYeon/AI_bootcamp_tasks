# set the LANGCHAIN_API_KEY environment variable (create key in settings)
from langchain import hub
import os
from dotenv import load_dotenv

load_dotenv()
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Langchain 라이브러리에서 원하는 프롬프트 댕겨오기

os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

prompt = hub.pull("rlm/rag-answer-hallucination")

# 프롬프트 텍스트만 추출
prompt_text = prompt.messages[0].prompt.template

# 댕겨온 프롬프트 template 저장
output_dir = "Prompts"
output_path = os.path.join(output_dir, "prompt3.txt")

# 프롬프트 데이터를 파일에 저장
with open(output_path, "w", encoding="utf-8") as file:
    file.write(prompt_text)

print(f"Prompt saved to {output_path}")

'''
1. 한국어 답변 프롬프트
2. 질문-답변 쌍 생성 프롬프트
3. 데이터 기반 문장여부 확인 프롬프트

'''