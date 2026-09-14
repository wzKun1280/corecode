from typing import Callable, Union
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from utils.logger_handler import logger

# ============ 下面四个是课程自定义占位装饰器（需要自己实现底层逻辑） ============
def wrap_tool_call(func):
    return func

def before_model(func):
    return func

def dynamic_prompt(func):
    return func

# 自定义ModelRequest类（课程内部对象）
class ModelRequest:
    def __init__(self, runtime):
        self.runtime = runtime

# 自定义AgentState
AgentState = dict


# ====================== 修复后的完整中间件代码（消除语法爆红） ======================
@wrap_tool_call
def monitor_tool(
        request,
        handler: Callable[[object], Union[ToolMessage, Command]],
) -> Union[ToolMessage, Command]:
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == "fill_context_for_report":   #只要调用这个工具就改为true
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime,
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
