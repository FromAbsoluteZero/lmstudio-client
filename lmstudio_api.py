#!/usr/bin/env python3
"""A small client and command line for LM Studio's native REST API (v1), standard library only.

It covers the two endpoints most scripts need:

    GET  /api/v1/models   the models LM Studio has downloaded, and which of them are loaded
    POST /api/v1/chat     send a message to a model and read its reply

Configuration comes from the environment, so nothing secret is written into a file:

    LM_STUDIO_URL    where the server listens                (default http://localhost:1234)
    LM_API_TOKEN     the API token, needed only when Require Authentication is on
    LM_STUDIO_MODEL  the model key to use                    (default: a loaded LLM, else the first LLM)

From the command line:

    python lmstudio_api.py models
    python lmstudio_api.py chat "Explain overfitting in two sentences."

From Python:

    import lmstudio_api as lms
    reply = lms.chat(lms.resolve_model(), "Explain overfitting in two sentences.")
    print(lms.reply_text(reply))
"""
import argparse, json, os, socket, sys, urllib.error, urllib.request

DEFAULT_URL = "http://localhost:1234"


class LMStudioError(RuntimeError):
    """The server could not be reached, or it answered with an error."""


def _request(method, path, body=None, base_url=None, token=None, timeout=300):
    base_url = (base_url or os.environ.get("LM_STUDIO_URL") or DEFAULT_URL).rstrip("/")
    token = os.environ.get("LM_API_TOKEN") if token is None else token
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace").strip()[:500]
        hint = ""
        if e.code in (401, 403):
            hint = (" The server rejected LM_API_TOKEN: check it has not been deleted in LM Studio"
                    " and has permission for this endpoint." if token else
                    " The server requires an API token: set LM_API_TOKEN.")
        raise LMStudioError(f"LM Studio answered {e.code} to {method} {path}.{hint} {detail}".rstrip()) from None
    except (TimeoutError, socket.timeout):
        # Connected, but no answer in time: a slow model load or a long reply.
        raise LMStudioError(f"LM Studio at {base_url} did not answer {method} {path} within "
                            f"{timeout}s.") from None
    except (urllib.error.URLError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        raise LMStudioError(
            f"Could not reach LM Studio at {base_url} ({reason}). Check the server is running; if it "
            f"is on another machine, turn on Serve on Local Network and set LM_STUDIO_URL to the "
            f"address it shows.") from None
    except json.JSONDecodeError:
        raise LMStudioError(f"{base_url} answered {method} {path} with something other than JSON; "
                            f"is LM_STUDIO_URL pointing at LM Studio?") from None


def list_models(base_url=None, token=None, timeout=30):
    """Every model LM Studio has downloaded: LLMs and embedding models, loaded or not."""
    return _request("GET", "/api/v1/models", base_url=base_url, token=token, timeout=timeout)["models"]


def pick_model(models):
    """The model to use when none is named: a loaded LLM if there is one, else the first LLM.

    With Just-in-Time Model Loading on, LM Studio loads an unloaded model on its first request."""
    llms = [m for m in models if m.get("type") == "llm"]
    if not llms:
        raise LMStudioError("LM Studio has no LLM downloaded. Download one in LM Studio first.")
    loaded = [m for m in llms if m.get("loaded_instances")]
    return (loaded or llms)[0]["key"]


def resolve_model(model=None, base_url=None, token=None, timeout=30):
    """The model named here, else LM_STUDIO_MODEL, else pick_model() over what is downloaded."""
    return (model or os.environ.get("LM_STUDIO_MODEL")
            or pick_model(list_models(base_url=base_url, token=token, timeout=timeout)))


def chat(model, message, system_prompt=None, base_url=None, token=None, timeout=300, **params):
    """Send one message to POST /api/v1/chat and return the parsed response.

    `message` is a string, or a list of input objects ({"type": "text", "content": ...} or
    {"type": "image", "data_url": ...}). Any other request field the endpoint accepts goes in
    `params` under its own name: temperature, max_output_tokens, store, previous_response_id,
    reasoning, context_length and so on. Fields left as None are not sent, so the server's
    defaults apply."""
    body = {"model": model, "input": message}
    if system_prompt is not None:
        body["system_prompt"] = system_prompt
    body.update({k: v for k, v in params.items() if v is not None})
    return _request("POST", "/api/v1/chat", body, base_url=base_url, token=token, timeout=timeout)


def reply_text(response):
    """The model's answer: the text of its "message" items, without reasoning or tool calls."""
    return "\n\n".join(item["content"] for item in response.get("output", [])
                       if item.get("type") == "message").strip()


def describe_stats(response):
    """One line of the response's token counts and speed, or "" when it carries none."""
    s = response.get("stats") or {}
    if not s:
        return ""
    line = (f"{s.get('input_tokens', '?')} tokens in, {s.get('total_output_tokens', '?')} out, "
            f"{s.get('tokens_per_second', 0):.1f} tokens/s")
    if "model_load_time_seconds" in s:
        line += f", model loaded in {s['model_load_time_seconds']:.1f}s"
    return line


def _print_models(models):
    for m in models:
        loaded = "loaded" if m.get("loaded_instances") else ""
        size = f"{m['size_bytes'] / 1e9:.1f} GB" if m.get("size_bytes") else ""
        print(f"  {m.get('type', '?'):<9} {loaded:<6}  {m['key']:<48} "
              f"{m.get('params_string') or '':<8} {size:>8}")
    if not models:
        print("  (none downloaded)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Talk to LM Studio's native REST API.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list the models LM Studio has downloaded")
    c = sub.add_parser("chat", help="send one message and print the reply")
    c.add_argument("message")
    c.add_argument("--model", help="model key (default: LM_STUDIO_MODEL, else a loaded LLM)")
    c.add_argument("--system", help="system prompt")
    c.add_argument("--temperature", type=float)
    c.add_argument("--max-tokens", type=int, dest="max_output_tokens")
    c.add_argument("--previous", dest="previous_response_id",
                   help="continue the chat that returned this response_id")
    args = p.parse_args(argv)
    try:
        if args.command == "models":
            _print_models(list_models())
            return 0
        model = resolve_model(args.model)
        r = chat(model, args.message, system_prompt=args.system, temperature=args.temperature,
                 max_output_tokens=args.max_output_tokens,
                 previous_response_id=args.previous_response_id)
    except LMStudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(reply_text(r))
    # Everything but the reply goes to stderr, so the reply alone can be piped or redirected.
    stats = describe_stats(r)
    print(f"\n[{r.get('model_instance_id', model)}{'; ' + stats if stats else ''}]", file=sys.stderr)
    if r.get("response_id"):
        print(f"[continue with --previous {r['response_id']}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
