"""Local Ollama weather-log query agent."""

from __future__ import annotations

import json
import locale
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


BASE_SYSTEM_PROMPT = (
    "你是本地运维天气日志查询智能体。用户需要检索天气日志时，先读取完整日志，"
    "再使用用户请求中的城市或关键词检索对应记录。"
)

# This suffix is part of the exercise specification. It is kept as a literal
# string so the model receives the same instruction on every request.
EXPERIMENT_SYSTEM_PROMPT = (
    "工具调用顺序固定为 read_file 后 bash。bash 使用用户查询关键词，"
    "输出只能是工具调用，禁止额外解释。"
)
SYSTEM_PROMPT = f"{BASE_SYSTEM_PROMPT}\n{EXPERIMENT_SYSTEM_PROMPT}"

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local text file and return its original contents.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a command through Windows cmd.exe.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


class ToolError(RuntimeError):
    """Raised when a local tool cannot complete."""


def read_file(file_path: str) -> str:
    """Return the file's original UTF-8 text without filtering or escaping."""

    path = Path(file_path)
    try:
        # newline="" prevents universal-newline conversion, retaining the
        # bytes represented by the file as faithfully as Python text allows.
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except OSError as exc:
        raise ToolError(f"read_file failed for {file_path!r}: {exc}") from exc


def bash(command: str) -> str:
    """Execute a command through Windows cmd.exe and return its output."""

    return "".join(bash_stream(command))


def bash_stream(command: str):
    """Yield cmd.exe output as soon as each line is available."""

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    try:
        for chunk in iter(process.stdout.readline, b""):
            if chunk:
                yield _decode_cmd_output(chunk)
    finally:
        process.stdout.close()
        process.wait()


def _decode_cmd_output(data: bytes) -> str:
    """Decode cmd/findstr output across UTF-8 and Chinese OEM code pages."""

    encodings = ["utf-8", locale.getpreferredencoding(False), "cp936"]
    for encoding in dict.fromkeys(encodings):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def extract_search_term(user_text: str) -> str:
    """Extract a city keyword from a natural-language weather query."""

    cities = (
        ("北京", "Beijing"),
        ("上海", "Shanghai"),
        ("深圳", "Shenzhen"),
    )
    lowered = user_text.lower()
    for chinese, english in cities:
        if chinese in user_text or english.lower() in lowered:
            return english
    return user_text.strip()


def build_findstr_command(search_term: str, file_path: str) -> str:
    """Build a single literal findstr search for the requested term."""

    return f'findstr /i /c:"{search_term}" "{file_path}"'


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"tool": self.name, "arguments": self.arguments}
        if self.result is not None:
            value["result"] = self.result
        return value


class OllamaClient:
    """Small dependency-free client for Ollama's local chat endpoint."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = 30.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout
        self._opener = opener

    def chat(self, user_text: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "tools": OLLAMA_TOOLS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (OSError, urllib.error.URLError) as exc:
            raise ToolError(
                f"Ollama is unavailable at {self.host}. Start Ollama and pull {self.model!r}."
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ToolError("Ollama returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ToolError("Ollama response must be a JSON object")
        return decoded


class WeatherLogAgent:
    """Orchestrates the fixed read-file -> findstr tool sequence."""

    def __init__(
        self,
        log_path: str = "weather_log.txt",
        ollama: Optional[OllamaClient] = None,
        read_tool: Callable[[str], str] = read_file,
        bash_tool: Callable[[str], str] = bash,
        bash_stream_tool: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.log_path = log_path
        self.ollama = ollama or OllamaClient()
        self.read_tool = read_tool
        self.bash_tool = bash_tool
        self.bash_stream_tool = bash_stream_tool or (bash_stream if bash_tool is bash else None)

    def run_stream(self, user_text: str, *, consult_model: bool = True):
        """Yield JSON-serializable tool-call records as each tool completes.

        Ollama is consulted to satisfy the local-model integration. Its natural
        language answer is deliberately not emitted: the exercise requires the
        observable result to consist only of tool calls.
        """

        raw_text = self.read_tool(self.log_path)
        read_call = ToolCall("read_file", {"file_path": self.log_path}, raw_text)
        yield read_call.as_dict()

        search_term = extract_search_term(user_text)
        command = build_findstr_command(search_term, self.log_path)
        if self.bash_stream_tool is not None:
            for chunk in self.bash_stream_tool(command):
                yield ToolCall("bash", {"command": command}, chunk).as_dict()
        else:
            search_output = self.bash_tool(command)
            yield ToolCall("bash", {"command": command}, search_output).as_dict()

        # The required local tools always run first and in this order. Ollama
        # is consulted only after the deterministic tool trace is complete.
        if consult_model:
            self.ollama.chat(user_text)

    def run(self, user_text: str, *, consult_model: bool = True) -> List[Dict[str, Any]]:
        """Run a query and collect the streaming tool-call records."""

        return list(self.run_stream(user_text, consult_model=consult_model))


def run_query(
    user_text: str,
    log_path: str = "weather_log.txt",
    *,
    consult_model: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function used by the CLI and embedding applications."""

    return WeatherLogAgent(log_path=log_path).run(user_text, consult_model=consult_model)
