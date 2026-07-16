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

1. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → repo `spaul571/ai-analytics-platform`, branch `main`, main file
   `app.py`.
3. **Advanced settings → Python version: 3.13.** The dependency pins in
   `requirements.txt` were verified on 3.13 and are not guaranteed elsewhere.
4. Deploy. It will build and then fail to answer questions — expected, there is
   no LLM behind it yet. Step 4 fixes that.

There is deliberately no `packages.txt`. Kaleido 1.x rasterises Plotly figures
by driving a real browser and Cloud's image has none, but asking apt for
`chromium` pulls Debian's own libpython into the container and the interpreter
segfaults on boot — the whole app, to serve four download buttons. Instead
`src/viz/browserless.py` redraws bar/line/scatter figures with matplotlib, so
the Task C4 exports keep their chart. Only the choropleth cannot be redrawn:
on Cloud its image buttons are hidden and its report prints without the map.
Locally, where Chrome exists, kaleido does the work and the fallback never runs.

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

## Starting it, every time

### 1. LM Studio, by hand

Load `google/gemma-4-e4b` and start the server on port 1234 (Developer tab →
Status: Running). Two settings to change first — both need an eject + reload:

- **Context Length: 131072 → 16384.** The KV cache at 131k reserves several GB
  of VRAM for prompts that never exceed ~2,300 tokens; the logs already show
  `failed to fit params to free device memory`. Dropping it frees VRAM for model
  layers and speeds up inference.
- **Enable Thinking: OFF.** Reasoning tokens add latency and can leak into the
  code that Phase 1 parses.

### 2. Everything else, with one command

```powershell
.\deploy\start-demo.ps1
```

It refuses to continue if LM Studio is not answering, starts the auth proxy,
opens the tunnel, waits for the public hostname to resolve, confirms an
unauthenticated request is refused with a 401, and then prints the exact secrets
block to paste into Streamlit Cloud.

The **token is reused across runs** — it is stored in `.env` and generated only
once — so `LLM_API_KEY` in the Cloud secrets stays valid forever.

### 3. Update one secret

**The hostname is new every single run.** A `trycloudflare.com` quick tunnel gets
a fresh random name each time `cloudflared` starts, so after every restart the
`LLM_BASE_URL` in the Cloud secrets points at a dead address until you update it.
That is the one manual step, and it cannot be avoided without a named tunnel on a
domain you own (`cloudflared tunnel create`) — worth doing if you have a domain,
not worth it if you do not.

Paste the printed block into **App settings → Secrets** and save. The app reboots
and answers.

## Stopping it

```powershell
.\deploy\stop-demo.ps1
```

Kills the proxy and the tunnel. LM Studio and the Cloud deployment are left
alone; the deployed app will simply have nothing to talk to until you start the
tunnel again.

Nothing here costs money while it is down, and nothing needs to be torn down in
the Cloud dashboard between demos.

---

## Failure modes, and what they look like

| Symptom in the deployed app                  | Cause                                    | Fix                                                          |
| -------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| Every question fails with a connection error | Tunnel restarted, hostname changed       | Update`LLM_BASE_URL` in Cloud secrets                      |
| Every question fails with a 401              | `LLM_API_KEY` ≠ `PROXY_TOKEN`       | Make them equal; restart proxy after changing`PROXY_TOKEN` |
| App loads, charts work, questions time out   | LM Studio not running or no model loaded | Load the model, restart the server                           |
| Reports download without their chart          | No browser in the container, and the figure is a map | Expected on Cloud. Bar/line/scatter fall back to matplotlib; the choropleth cannot |
| App boots but no data                        | `data/superstore.csv` not committed    | It is committed on purpose — see`.gitignore`              |

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
