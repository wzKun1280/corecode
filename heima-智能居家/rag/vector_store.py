from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import get_embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    def __init__(self):
        # Chroma初始化
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # 向量集合名称
            embedding_function=get_embed_model(),  # 向量模型，用来把文字转为向量
            persist_directory=chroma_conf["persist_directory"],  # 本地文件夹持久化向量数据
        )

        self.spliter = RecursiveCharacterTextSplitter(  # 文本分割器
            chunk_size=chroma_conf["chunk_size"],  # 每一块文本最大字符长度
            chunk_overlap=chroma_conf["chunk_overlap"],  # 分片重叠字符串
            separators=chroma_conf["separators"],  # 分割符号列表
            length_function=len,  # 使用字符长度判断块大小
        )

    # 向量库调用 `.as_retriever()` 转为**检索器 Retriever**
    def get_retriever(self):  # 获取检索器
        # 每次问答召回相似度最高的 k 条文本块
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    # 知识库加载主函数
    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):  # 判断文件是否已经入库 防止重复向量
            md5_file_path = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_file_path):
                # 创建空文件，with写法更安全
                with open(md5_file_path, "w", encoding="utf-8") as f:
                    pass
                return False  # md5 没处理过

            with open(md5_file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5已存在
            return False  # md5不存在

        def save_md5_hex(md5_for_check: str):  # 文件入库后，追加写入MD5字符串
            # 增加类型强校验，杜绝function + str报错
            if not isinstance(md5_for_check, str):
                logger.warning(f"md5值类型异常，不是字符串：{md5_for_check}")
                return
            # a 的意思是追加模式，不会覆盖原有记录
            md5_file_path = get_abs_path(chroma_conf["md5_hex_store"])
            with open(md5_file_path, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            # 判断文件后缀，调用不同的工具读取文本，返回 `Document` 对象列表；不支持的文件返回空列表
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        # 遍历全部合法文件的主循环
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5，强制调用函数（带括号）
            md5_result = get_file_md5_hex(path)
            logger.debug(f"文件 {path} 的md5结果：{md5_result}，类型：{type(md5_result)}")

            # 增加合法性判断，如果md5不是字符串，直接跳过该文件
            if not isinstance(md5_result, str):
                logger.error(f"[加载知识库]{path} MD5计算失败，跳过")
                continue

            if check_md5_hex(md5_result):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:  # 文档解析、切分、存入向量库
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_result)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
