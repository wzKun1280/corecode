import os
import certifi
from typing import TypedDict, Annotated, Union, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
import operator

os.environ["SSL_CERT_FILE"] = certifi.where()

from model.factory import get_chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from utils.logger_handler import logger

# ====================== 1. 简化State定义（兼容低版本Python） ======================
class AgentState(TypedDict):
    messages: Annotated[list[Any], operator.add]
    report: bool

# ====================== 2. 工具列表 ======================
tool_list = [rag_summarize, get_weather, get_user_location, get_user_id,
             get_current_month, fetch_external_data, fill_context_for_report]
chat_model = get_chat_model().bind_tools(tool_list)

# ====================== 通用工具：安全读取消息content ======================
# 字典格式 / LangChain 消息对象，安全提取文本内容
def get_msg_content(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    else:
        return getattr(msg, "content", "")

# ====================== 3. 带日志监控的ToolNode ======================
def monitored_tool_node(state: AgentState):
    tool_node = ToolNode(tool_list)
    # LangGraph 官方内置节点，自动解析`tool_calls`并执行对应函数
    # 取出对话列表最后一条消息，这条消息就是上一轮大模型返回的结果，里面可能携带 `tool_calls`
    last_msg = state["messages"][-1]
    # 判断是否为消息对象，并且携带tool_calls
    # 打印日志：记录即将执行的工具名称和参数
    if not isinstance(last_msg, dict) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for call in last_msg.tool_calls:
            logger.info(f"[tool monitor]执行工具：{call['name']}")
            logger.info(f"[tool monitor]传入参数：{call['args']}")
    result = tool_node.invoke(state)
    # 特殊业务逻辑：如果调用fill_context_for_report，打开报告开关 report=True
    for msg in result["messages"]:  # 遍历工具执行完毕之后返回的全部消息
        if getattr(msg, "name", None) == "fill_context_for_report":
            result["report"] = True # 后续模型切换报告专用提示词
            logger.info("[tool monitor]fill_context_for_report调用成功，开启报告模式")
    return result

# ====================== 4. LLM节点：打印日志 + 动态切换Prompt ======================
def call_model(state: AgentState):
    msg_count = len(state["messages"])   # 统计当前对话消息总数，打印日志
    logger.info(f"[log_before_model]即将调用模型，带有{msg_count}条消息。")
    last_msg = state["messages"][-1]
    last_content = get_msg_content(last_msg)  # 拿到对话最后一条消息
    logger.debug(f"[log_before_model]{type(last_msg).__name__} | {last_content.strip()}")
    # 根据report标记动态切换系统提示词
    # report = False → 读取普通对话 prompt
    if state.get("report", False):
        sys_content = load_report_prompts()
    else:
        sys_content = load_system_prompts()

    from langchain_core.messages import SystemMessage
    system_msg = SystemMessage(content=sys_content)
    # 拼接完整对话上下文：**系统提示词放在最前面，后面拼接历史全部对话消息**。
    all_messages = [system_msg] + state["messages"]
    # 请求大模型
    response = chat_model.invoke(all_messages)
    return {"messages": [response]}

# ====================== 5. 路由判断是否调用工具 ======================
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not isinstance(last_message, dict) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# ====================== 6. 构建流程图 ======================
def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", monitored_tool_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
    graph_builder.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

# ====================== 7. Agent封装类 ======================
class ReactAgent:
    def __init__(self):
        self.agent = build_graph()
        self.config = {"configurable": {"thread_id": "heima_agent_001"}}

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ],
            "report": False
        }
        for chunk in self.agent.stream(input_dict, config=self.config, stream_mode="values"):
            msg_list = chunk.get("messages", [])
            if not msg_list:
                continue
            latest_message = msg_list[-1]
            content = get_msg_content(latest_message)
            if content:
                yield content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
