import os
from langchain_openai import ChatOpenAI
from langchain.agents import tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

assert load_dotenv()

# 确保你已经设置了 OpenAI API 密钥
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# --- 新增部分：自定义回调处理器 ---
class PrintMessagesCallback(BaseCallbackHandler):
    """一个简单的回调处理器，用于在聊天模型调用开始时打印消息。"""
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """当聊天模型即将被调用时触发。"""
        print("\n--- 发送给 LLM 的完整消息列表 ---")
        for role_messages in messages:
            for msg in role_messages:
                print(f"[{msg.type.upper()}]: {msg.content}")
                print("-------------------------------------------------\n")

# 定义一个简单的工具
@tool
def get_weather(city: str) -> str:
    """当需要查询天气时，调用此工具。输入参数是城市名称。"""
    if city == "北京":
        return "北京今天晴，25摄氏度。"
    elif city == "上海":
        return "上海今天有雨，20摄氏度。"
    else:
        return f"抱歉，我没有 {city} 的天气信息。"

# 1. 初始化 LLM 模型
llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-nano")

# 2. 定义工具列表
tools = [get_weather]

# 3. 初始化记忆模块
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. 初始化 Agent
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# 5. 创建回调处理器的实例
callbacks = [PrintMessagesCallback()]

print("--- 第一次查询 (无历史记录) ---")
# 在调用时传入 callbacks
response1 = agent_chain.invoke(
    {"input": "你好，我想知道北京的天气怎么样？"},
    config={"callbacks": callbacks}
)
print(f"AI 回答: {response1['output']}")

print("\n" + "="*50 + "\n")

print("--- 第二次查询 (利用了第一次的对话历史) ---")
# 再次传入 callbacks
response2 = agent_chain.invoke(
    {"input": "那上海呢？"},
    config={"callbacks": callbacks}
)
print(f"AI 回答: {response2['output']}")