import os
import hashlib

from IPython.utils import path
from sqlalchemy.util import md5_hex

from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

#获取文件的MD5十六进制的字符串
def get_file_md5_hex(filepath: str):

    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[md5]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()  #创建一个空的MD5摘要计算器

    chunk_size = 4096  #4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:   #二进制模式读取，不能用文本模式r
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """ 解释一些:= 海象运算符 新出的
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_obj = md5_obj.hexdigest()  #将二进制MD5转换成32位小写十六进制字符串
            return md5_obj
    except Exception as e:  #文件权限不足、被占用、磁盘损坏等，记录日志返回None
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):  #遍历文件夹，筛选指定后缀的文件
    files = []
    # path：要扫描的文件夹路径
    # `allowed_types: tuple[str]`：允许的后缀元组，示例：`(".pdf",".txt")`
    if not os.path.isdir(path):   #判断路径是否我呢见加
        logger.error(f"[lisrdir_with_allowed_type]{path}不是文件夹")
        return allowed_types    # 如果不是文件夹，函数直接返回了`allowed_types`（后缀元组），正常业务应该返回空元组 `return ()`

    for f in os.listdir(path):  #获取文件夹下所有子文件/子文件夹名称（不带完整路径）
        if f.endswith(allowed_types):   #判断文件夹是否以指定后缀结尾
            files.append(os.path.join(path,f))   #添加文件到指定位置下面

    return tuple(files)  # 返回元组？ 因为元组对外输出只读数据（tuple） 后续还要增删改则使用list

def pdf_loader(filepath: str, passwd = None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()   #.load 执行读取解析
    # 返回值list[Document]: 一个PDF每一页对应一个Document对象 包含当前页面的文本内容、元数据等内容

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
