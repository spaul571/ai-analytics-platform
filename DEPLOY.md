# Deployment

The brief asks for two things that pull against each other:

- **M6** — *"Live demo from GitHub deployment."*
- **§7.3** — *"all AI calls must be live during the demo."*

The model runs locally in LM Studio. A cloud-hosted frontend cannot reach
`localhost:1234` on a laptop, so a naive Streamlit Cloud deploy would have no
LLM at all, and faking the calls is explicitly prohibited. The resolution is to
deploy the frontend from GitHub and give the local model a public address:

```
  browser
     |
     v
  Streamlit Community Cloud          <- deployed from this GitHub repo (M6)
     |   LLM_BASE_URL secret
     v
  Cloudflare edge (public hostname)
     ^
     |   outbound tunnel, no open inbound ports
  cloudflared        ] both run on the machine
  deploy/llm_proxy.py]  that has the GPU
     |   127.0.0.1:1234
     v
  LM Studio - Gemma 4 E4B            <- the AI calls are genuinely live (§7.3)
```

Two properties of this shape matter:

**The tunnel is outbound.** `cloudflared` dials out to Cloudflare; nothing dials
in to you. No port forwarding, no firewall exception — and university wifi that
isolates clients from each other cannot break it, because Streamlit Cloud talks
to Cloudflare, not to your laptop. The laptop only needs ordinary internet.

**The proxy is not optional.** LM Studio's server has no authentication: anyone
who reaches port 1234 can run inference and enumerate your models. That is
harmless on localhost and unacceptable once the port has a public hostname.
`deploy/llm_proxy.py` rejects any request without the right bearer token. The
OpenAI SDK already sends `LLM_API_KEY` as `Authorization: Bearer <key>`, so
setting `LLM_API_KEY == PROXY_TOKEN` is the whole integration — no app code
changes.

---

## One-time setup

### 1. Install cloudflared

```powershell
winget install --id Cloudflare.cloudflared
```

### 2. Generate a token

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep it. It goes in two places: `PROXY_TOKEN` on your machine, and
`LLM_API_KEY` in the Streamlit Cloud secrets.

### 3. Deploy the frontend

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → repo `spaul571/ai-analytics-platform`, branch `main`, main file
   `app.py`.
3. **Advanced settings → Python version: 3.13.** The dependency pins in
   `requirements.txt` were verified on 3.13 and are not guaranteed elsewhere.
4. Deploy. It will build and then fail to answer questions — expected, there is
   no LLM behind it yet. Step 4 fixes that.

`packages.txt` installs Chromium into the container. Kaleido 1.x drives a real
browser to rasterise Plotly figures, and Cloud's image has none; without it the
PDF/PNG/SVG exports in Task C4 throw at runtime while the rest of the app looks
fine.

### 4. Point the app at your machine

Start the two local processes (see *Every demo* below), take the tunnel
hostname it prints, then in the Streamlit Cloud dashboard go to
**App settings → Secrets** and paste:

```toml
LLM_BASE_URL = "https://<hostname>.trycloudflare.com/v1"
LLM_MODEL    = "google/gemma-4-e4b"
LLM_API_KEY  = "<the token from step 2>"
```

The `/v1` suffix is required — the OpenAI SDK appends `/chat/completions` to it.

Saving secrets reboots the app. Ask it a question; you should see the request
arrive in the proxy's console.

---

## Every demo

Three processes, in this order. Keep all three windows open.

**1. LM Studio.** Load `google/gemma-4-e4b`, start the server on port 1234.
Before the demo, change two settings — both need an eject + reload:

- **Context Length: 131072 → 16384.** The KV cache at 131k reserves several GB
  of VRAM for prompts that never exceed ~2,300 tokens; the logs already show
  `failed to fit params to free device memory`. Dropping it frees VRAM for model
  layers and speeds up inference.
- **Enable Thinking: OFF.** Reasoning tokens add latency and can leak into the
  code that Phase 1 parses.

**2. The auth proxy.**

```powershell
$env:PROXY_TOKEN = "<the token from step 2>"
.venv\Scripts\python.exe -m uvicorn deploy.llm_proxy:app --port 1235
```

It refuses to start without `PROXY_TOKEN` — an unauthenticated proxy would
defeat its own purpose, so there is deliberately no default.

**3. The tunnel.**

```powershell
cloudflared tunnel --url http://localhost:1235
```

It prints a `https://<random>.trycloudflare.com` hostname.

> **A quick tunnel gets a new random hostname every restart.** If you restart
> `cloudflared`, you must update `LLM_BASE_URL` in the Streamlit secrets or the
> deployed app is pointing at a dead address. To get a hostname that survives
> restarts you need a named tunnel on a domain you control
> (`cloudflared tunnel create`), which is worth doing if you have a domain and
> not worth it if you do not.

### Check it end to end before you present

```powershell
# 1. tunnel is alive (no token needed, reveals nothing)
curl https://<hostname>.trycloudflare.com/healthz

# 2. an unauthenticated call is refused - should print 401
curl -o /dev/null -w "%{http_code}" https://<hostname>.trycloudflare.com/v1/models

# 3. an authenticated call reaches the model - should list gemma
curl -H "Authorization: Bearer <token>" https://<hostname>.trycloudflare.com/v1/models
```

Then open the Streamlit Cloud URL and ask one real question. If the answer comes
back, every hop in the diagram works.

---

## Failure modes, and what they look like

| Symptom in the deployed app | Cause | Fix |
|---|---|---|
| Every question fails with a connection error | Tunnel restarted, hostname changed | Update `LLM_BASE_URL` in Cloud secrets |
| Every question fails with a 401 | `LLM_API_KEY` ≠ `PROXY_TOKEN` | Make them equal; restart proxy after changing `PROXY_TOKEN` |
| App loads, charts work, questions time out | LM Studio not running or no model loaded | Load the model, restart the server |
| Charts render, exports throw | Chromium missing in the container | `packages.txt` is committed; check it deployed |
| App boots but no data | `data/superstore.csv` not committed | It is committed on purpose — see `.gitignore` |

## Contingency

If the tunnel dies mid-demo, the app is dead until the hostname is updated,
which is a bad thing to do in front of an assessor. Two mitigations, in order of
preference:

1. **Run the app locally as well.** `streamlit run app.py` against
   `http://localhost:1234/v1` needs no tunnel and no proxy. Have this window
   open and ready. The GitHub deployment satisfies M6; falling back to the local
   instance for the last few questions is honest and costs you nothing if you
   say what you are doing and why.
2. **Have the hostname update ready to paste.** Restarting `cloudflared` and
   editing one secret takes about 30 seconds if you are not searching for the
   dashboard while people watch.
