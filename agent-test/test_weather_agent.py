import json
import unittest

from weather_agent import (
    OLLAMA_TOOLS,
    OllamaClient,
    WeatherLogAgent,
    build_findstr_command,
    extract_search_term,
)


class WeatherAgentTests(unittest.TestCase):
    def test_search_term_comes_from_user_query(self):
        self.assertEqual(extract_search_term("只查询北京天气"), "Beijing")
        self.assertEqual(
            build_findstr_command("Beijing", "weather_log.txt"),
            'findstr /i /c:"Beijing" "weather_log.txt"',
        )

    def test_agent_runs_tools_in_order_without_model(self):
        events = []

        def fake_read(path):
            events.append(("read_file", path))
            return "cloudy 20C"

        def fake_bash(command):
            events.append(("bash", command))
            return "weather_log.txt:cloudy 20C"

        agent = WeatherLogAgent(read_tool=fake_read, bash_tool=fake_bash, ollama=object())
        result = agent.run("query", consult_model=False)
        self.assertEqual(
            events,
            [("read_file", "weather_log.txt"), ("bash", 'findstr /i /c:"query" "weather_log.txt"')],
        )
        self.assertEqual(result[0]["tool"], "read_file")
        self.assertEqual(result[1]["tool"], "bash")
        json.dumps(result, ensure_ascii=False)

    def test_agent_yields_bash_chunks(self):
        agent = WeatherLogAgent(
            read_tool=lambda _: "line one\nline two\n",
            bash_stream_tool=lambda _: iter(["line one\r\n", "line two\r\n"]),
            ollama=object(),
        )
        result = list(agent.run_stream("查询", consult_model=False))
        self.assertEqual(result[0]["tool"], "read_file")
        self.assertEqual([call["result"] for call in result[1:]], ["line one\r\n", "line two\r\n"])

    def test_ollama_payload_uses_local_model_and_tools(self):
        requests = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":""}}'

        def opener(request, timeout):
            requests.append((request, timeout))
            return Response()

        OllamaClient(model="test-model", opener=opener).chat("query")
        payload = json.loads(requests[0][0].data.decode("utf-8"))
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual([tool["function"]["name"] for tool in payload["tools"]], ["read_file", "bash"])


if __name__ == "__main__":
    unittest.main()
