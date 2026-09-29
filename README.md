# lmstudio-client

A small Python client and command line for [LM Studio](https://lmstudio.ai)'s native REST API
(`/api/v1/*`, LM Studio 0.4.0 or newer). One file, standard library only: nothing to install.

It covers the two endpoints most scripts need:

| Endpoint | Function | Command |
|---|---|---|
| `GET /api/v1/models` | `list_models()` | `python lmstudio_api.py models` |
| `POST /api/v1/chat` | `chat()` | `python lmstudio_api.py chat "..."` |

## Setting up LM Studio

1. Download a model in LM Studio. With **Just-in-Time Model Loading** on, the client loads it on
   first use; with it off, load the model in LM Studio first.
2. Start the server from LM Studio's developer page. It listens on port 1234 unless you change it.
3. If the client runs on a different machine from LM Studio, turn on **Serve on Local Network** and
   use the address LM Studio shows.
4. If **Require Authentication** is on, create a token under **Manage Tokens**. LM Studio shows it
   once, when you create it.

## Configuration

Three environment variables, all optional:

| Variable | Default | What it is |
|---|---|---|
| `LM_STUDIO_URL` | `http://localhost:1234` | the server's address; for another machine on your network, something like `http://192.168.1.20:1234` |
| `LM_API_TOKEN` | none | the API token, needed only when Require Authentication is on |
| `LM_STUDIO_MODEL` | a loaded LLM, else the first LLM downloaded | the model's key, as `models` lists it |

```bash
export LM_STUDIO_URL=http://192.168.1.20:1234
export LM_API_TOKEN=...          # Windows PowerShell: $env:LM_API_TOKEN = "..."
```

**Keep the token out of files you commit, and out of chats.** Set it in your terminal as above. If
you keep it in a `.env` file instead, `.gitignore` already excludes that. A token that has been
pasted anywhere public, or into a chat, should be deleted in LM Studio's Manage Tokens and replaced.

## Command line

```bash
python lmstudio_api.py models          # checks the connection, and lists what is downloaded
python lmstudio_api.py chat "Explain overfitting in two sentences."
python lmstudio_api.py chat "And underfitting?" --previous resp_...   # continue that chat
python lmstudio_api.py chat "Summarise this" --model example/chat-model --system "Be brief." --temperature 0.2 --max-tokens 200
```

`chat` prints the reply alone on standard output, so it can be piped or redirected; the token counts,
speed and the `response_id` to continue from go to standard error. LM Studio stores the chat on its
side, so a follow-up with `--previous` sends only the new message.

## From Python

```python
import lmstudio_api as lms

model = lms.resolve_model()        # LM_STUDIO_MODEL, else a loaded LLM; or pass a key
reply = lms.chat(model, "Explain overfitting in two sentences.",
                 system_prompt="You are a patient teacher.", temperature=0, store=False)
print(lms.reply_text(reply))       # the answer, without reasoning or tool-call items
print(lms.describe_stats(reply))   # "120 tokens in, 18 out, 41.5 tokens/s"
```

Any other request field `/api/v1/chat` accepts goes to `chat()` under its own name:
`max_output_tokens`, `previous_response_id`, `reasoning`, `context_length`, `top_p`, `integrations`
and so on. Fields left as `None` are not sent, so LM Studio's defaults apply. `message` can also be
a list of input objects, for images:
`[{"type": "text", "content": "Describe this"}, {"type": "image", "data_url": "data:image/png;base64,..."}]`.

Every failure raises `lmstudio_api.LMStudioError` with a message saying what to check: a missing or
refused token, a server that cannot be reached, a timeout, or a reply that is not JSON. The token is
never included in an error message.

## Tests

```bash
python tests/test_lmstudio_api.py      # or: pytest
```

No LM Studio is needed: a fake server on 127.0.0.1 stands in for it, answering with the response
shapes LM Studio documents, and the tests check what the client sends, how it reads the reply, and
how it reports a refused token or an unreachable server.

Python 3.10 or newer.

## Licence

MIT. See `LICENSE`.
