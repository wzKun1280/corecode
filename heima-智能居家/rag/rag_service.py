# 用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回答
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import get_chat_model
# 这是**RAG 摘要问答服务类**，完整实现经典检索增强生成流程：
# 用户问题 → 向量库检索文档 → 拼接参考资料上下文 → 送入Prompt模板 → 大模型生成总结回答`

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()  # 向量库服务实例
        self.retriever = self.vector_store.get_retriever() # 获取检索器，用来根据 query 召回相似文档
        self.prompt_text = load_rag_prompts() # 读取配置文件中的 RAG 提示字符串
        self.prompt_template = PromptTemplate.from_template(self.prompt_text) # 把原始字符串包装成 LangChain 模板对象，支持变量填充
        self.model = get_chat_model() # 获取对话大模型
        self.chain = self._init_chain() # 组装完整调用链

    def _init_chain(self):
        # 组装链式执行管道，执行顺序从左到右
        # `prompt_template`：接收字典 `{"input":"问题","context":"拼接后的资料"}
        #  `print_prompt`：拦截打印完整 prompt，原样向下传递
        #  `self.model`：把 prompt 发给大模型，返回结构化消息对象
        # `StrOutputParser()`：提取模型 content，输出纯文本字符串
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain


    # 输入查询文本，调用向量检索，返回一批`Document`文档对象列表
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        # 检索匹配文档
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        # 遍历检索出来的文档 手动拼接成一段完整字符串`context`，给每篇资料增加编号标识
        for doc in context_docs:   # 循环拼接上下文，返回大模型总结后的回答字符串
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 启动 LangChain 管道，管道内部严格从左往右执行
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))




