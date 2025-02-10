import logging
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from pyngrok import ngrok

# 로그 설정
logging.basicConfig(level=logging.INFO)

# FastAPI 앱 생성
app = FastAPI()

# OpenAI 기반 LLM 설정
llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

# 정확한 프롬프트 템플릿
custom_prompt = PromptTemplate(
    input_variables=["history", "input"],  # 정확한 변수명 사용
    template= """

{history}

사용자의 질문: {input}
AI의 답변:"""
)

# 사용자별 대화 히스토리를 저장할 전역 변수
conversation_store = {}

def get_conversation_chain(user_id: str):
    """사용자별로 ConversationChain을 유지하는 함수"""
    if user_id not in conversation_store:
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)  # 정확한 메모리 키 설정
        conversation_store[user_id] = ConversationChain(llm=llm, memory=memory, prompt=custom_prompt)
    return conversation_store[user_id]

class QueryRequest(BaseModel):
    question: str
    user_id: str  

@app.post("/ask")
async def ask_question(request: QueryRequest):
    # 요청에서 user_id 추출하여 ConversationChain 가져오기
    conversation = get_conversation_chain(request.user_id)

    try:
        # AI 응답 생성
        response = conversation.predict(input=request.question)

    except Exception as e:
        logging.error(f"🔴 LLM Error: {e}")
        response = "⚠️ AI 응답을 생성하는 데 문제가 발생했습니다."

    return {
        "answer": response,
        "history": conversation.memory.load_memory_variables({})["history"]
    }

if __name__ == "__main__":
    import uvicorn
    from threading import Thread

    # ngrok을 실행하여 외부에서 접근 가능하도록 함
    public_url = ngrok.connect(8000).public_url
    print(f"🔗 Public URL: {public_url}")

    # FastAPI 서버 실행
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    server_thread = Thread(target=run_server)
    server_thread.start()
