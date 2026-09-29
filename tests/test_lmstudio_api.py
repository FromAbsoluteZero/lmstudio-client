"""lmstudio_api checked against a fake server: what it sends, how it reads a reply, and how it
reports a refused token or an unreachable server.

No LM Studio is needed. A small HTTP server on 127.0.0.1 stands in for it, answers with the
response shapes LM Studio's documentation gives for /api/v1/models and /api/v1/chat, and records
every request it receives.

Run:  python tests/test_lmstudio_api.py
  or: pytest
"""
import contextlib, io, json, os, socket, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lmstudio_api as api   # noqa: E402

EMBEDDING = {"type": "embedding", "key": "example/embedding-model",
             "loaded_instances": [{"id": "example/embedding-model", "config": {"context_length": 2048}}]}
UNLOADED = {"type": "llm", "key": "example/small-model", "params_string": "3B",
            "size_bytes": 2_100_000_000, "loaded_instances": []}
LOADED = {"type": "llm", "key": "example/chat-model", "params_string": "4B", "size_bytes": 2_500_000_000,
          "loaded_instances": [{"id": "example/chat-model", "config": {"context_length": 4096}}]}
MODELS = {"models": [EMBEDDING, UNLOADED, LOADED]}

ANSWER = "A model overfits when it learns the noise in its training data along with the signal."
CHAT_REPLY = {
    "model_instance_id": "example/chat-model",
    "output": [{"type": "reasoning", "content": "Define it, then give the consequence."},
               {"type": "message", "content": ANSWER}],
    "stats": {"input_tokens": 120, "total_output_tokens": 18, "reasoning_output_tokens": 9,
              "tokens_per_second": 41.5, "time_to_first_token_seconds": 0.2},
    "response_id": "resp_0123456789abcdef",
}


class FakeLMStudio:
    """A stand-in server. With `token` set, it refuses any request not carrying that token."""

    def __init__(self, token=None):
        self.token, self.requests = token, []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _answer(self):
                n = int(self.headers.get("Content-Length") or 0)
                owner.requests.append({"method": self.command, "path": self.path,
                                       "auth": self.headers.get("Authorization"),
                                       "body": json.loads(self.rfile.read(n)) if n else None})
                if owner.token and self.headers.get("Authorization") != f"Bearer {owner.token}":
                    return self._send(401, {"error": {"message": "Invalid or missing API token"}})
                if (self.command, self.path) == ("GET", "/api/v1/models"):
                    return self._send(200, MODELS)
                if (self.command, self.path) == ("POST", "/api/v1/chat"):
                    return self._send(200, CHAT_REPLY)
                self._send(404, {"error": {"message": "not found"}})

            do_GET = do_POST = _answer

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def chats(self):
        return [r["body"] for r in self.requests if r["path"] == "/api/v1/chat"]


@contextlib.contextmanager
def environment(**values):
    """Set (or, with None, remove) environment variables for the duration of a block."""
    saved = {k: os.environ.get(k) for k in values}
    for k, v in values.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def raises(fn):
    """The LMStudioError fn raises, or an AssertionError if it raises none."""
    try:
        fn()
    except api.LMStudioError as e:
        return str(e)
    raise AssertionError("expected LMStudioError")


def test_models_request_carries_the_bearer_token():
    with FakeLMStudio(token="test-value-123") as s:
        models = api.list_models(base_url=s.url, token="test-value-123")
    assert [m["key"] for m in models] == [m["key"] for m in MODELS["models"]]
    assert s.requests[0]["method"] == "GET" and s.requests[0]["auth"] == "Bearer test-value-123"


def test_no_authorization_header_without_a_token():
    with FakeLMStudio() as s:
        api.list_models(base_url=s.url + "/", token="")      # a trailing slash is tolerated
    assert s.requests[0]["path"] == "/api/v1/models" and s.requests[0]["auth"] is None


def test_chat_sends_only_the_fields_given():
    with FakeLMStudio() as s:
        r = api.chat("example/chat-model", "hi", system_prompt=None, base_url=s.url, token="",
                     temperature=0, max_output_tokens=None, store=False)
    # temperature=0 is falsy but deliberate, so it is sent; the None fields are not.
    assert s.chats() == [{"model": "example/chat-model", "input": "hi", "temperature": 0, "store": False}]
    assert r == CHAT_REPLY


def test_reply_text_keeps_only_the_message():
    assert api.reply_text(CHAT_REPLY) == ANSWER
    assert api.reply_text({"output": []}) == ""
    assert api.describe_stats(CHAT_REPLY) == "120 tokens in, 18 out, 41.5 tokens/s"
    assert api.describe_stats({"output": []}) == ""


def test_pick_model_prefers_a_loaded_llm():
    assert api.pick_model(MODELS["models"]) == "example/chat-model"      # not the loaded embedding model
    assert api.pick_model([EMBEDDING, UNLOADED]) == "example/small-model"
    assert "no LLM" in raises(lambda: api.pick_model([EMBEDDING]))
    with environment(LM_STUDIO_MODEL="named/model"):
        assert api.resolve_model() == "named/model"               # named: no request made
    assert api.resolve_model("given/model") == "given/model"


def test_a_refused_token_is_explained_and_never_echoed():
    with FakeLMStudio(token="the-right-one") as s:
        wrong = raises(lambda: api.list_models(base_url=s.url, token="a-wrong-value-456"))
        missing = raises(lambda: api.list_models(base_url=s.url, token=""))
    assert "401" in wrong and "rejected LM_API_TOKEN" in wrong
    assert "a-wrong-value-456" not in wrong and "the-right-one" not in wrong
    assert "401" in missing and "set LM_API_TOKEN" in missing


def test_an_unreachable_server_names_the_address():
    with socket.socket() as sock:                # a port that was free a moment ago, now closed
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    msg = raises(lambda: api.list_models(base_url=f"http://127.0.0.1:{port}", token="", timeout=5))
    assert f"http://127.0.0.1:{port}" in msg and "LM_STUDIO_URL" in msg


def run_cli(server, *argv, token=None):
    out, err = io.StringIO(), io.StringIO()
    with environment(LM_STUDIO_URL=server.url, LM_API_TOKEN=token, LM_STUDIO_MODEL=None), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = api.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_cli_models_and_chat():
    with FakeLMStudio(token="cli-value") as s:
        code, out, _ = run_cli(s, "models", token="cli-value")
        assert code == 0 and "example/chat-model" in out and "loaded" in out
        code, out, err = run_cli(s, "chat", "What is overfitting?", "--temperature", "0.2",
                                 "--previous", "resp_earlier", token="cli-value")
    # The reply alone on stdout, so it can be piped; the rest on stderr.
    assert code == 0 and out.strip() == ANSWER
    assert "41.5 tokens/s" in err and "--previous resp_0123456789abcdef" in err
    body = s.chats()[-1]
    assert body == {"model": "example/chat-model", "input": "What is overfitting?", "temperature": 0.2,
                    "previous_response_id": "resp_earlier"}


def test_cli_reports_errors_on_stderr_and_exits_nonzero():
    with FakeLMStudio(token="needed") as s:
        code, out, err = run_cli(s, "models")
    assert code == 1 and out == "" and err.startswith("error:") and "set LM_API_TOKEN" in err


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok    {t.__name__}")
    print(f"{len(tests)} checks passed")
