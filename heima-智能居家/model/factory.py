# 工厂类
from abc import ABC, abstractmethod
from typing import Optional,Union
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf

# 定义一个抽象类
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        return ChatTongyi(model=rag_conf["chat_model_name"],
        dashscope_api_key=rag_conf.get("dashscope_api_key", "")
                          )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"],
        dashscope_api_key=rag_conf.get("dashscope_api_key", ""))

def get_chat_model():
    return ChatModelFactory().generator()

def get_embed_model():
    return EmbeddingsFactory().generator()

# chat_model = ChatModelFactory().generator()
# embed_model = EmbeddingsFactory().generator()
