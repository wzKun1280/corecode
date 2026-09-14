import logger

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"]) #去 yml 字典读取配置值 `prompts/main_prompt.txt` 然后拼接成完整绝对路径
    except KeyError as e:    # yml 如果漏写这个配置，捕获异常打日志
        print(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        print(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        print(f"[load_report_prompts]yaml配置缺失：{str(e)}")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


if __name__ == '__main__':
    print(load_rag_prompts())

