# 本地运维天气日志查询智能体

这是一个使用本地 Ollama 的最小 Python 命令行实现。它固定执行以下顺序：

1. 读取 `weather_log.txt` 的完整原始文本；
2. 从用户请求中提取城市或关键词，构造 `findstr /i /c:"关键词" "文件路径"`；
3. 通过 Windows `cmd.exe` 执行该命令；
4. 以 JSON Lines 流式输出工具调用及结果；`bash` 的每一行结果会立即输出。

## 运行

本机 Ollama 已检测到 `qwen3:8b`，直接运行即可：

```powershell
python .\main.py "查询今天的天气日志"
```

可用环境变量 `OLLAMA_MODEL` 和 `OLLAMA_HOST` 覆盖模型与服务地址。只验证本地工具链时，可跳过模型请求：

```powershell
python .\main.py --offline "查询天气"
```

`read_file` 保留文件换行；例如“只查询北京天气”会检索 `Beijing`，只返回北京记录。
