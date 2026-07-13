"""Authenticating reverse proxy that sits in front of LM Studio.

LM Studio's OpenAI-compatible server has no authentication: anything that can
reach port 1234 can run inference and enumerate loaded models. That is fine on
localhost and unacceptable once a Cloudflare Tunnel gives the port a public
hostname, so this proxy is what the tunnel actually exposes.

It rejects any request whose Authorization header does not carry PROXY_TOKEN,
then forwards the rest verbatim to LM Studio.

    browser -> Streamlit Cloud -> Cloudflare edge -> this proxy -> LM Studio

The application code needs no change to work through it. The OpenAI SDK already
sends its api_key as `Authorization: Bearer <key>`, which LM Studio ignores, so
setting LLM_API_KEY to the same value as PROXY_TOKEN is the entire integration.

Run it with:

    PROXY_TOKEN=<secret> python -m uvicorn deploy.llm_proxy:app --port 1235

It binds to localhost only. cloudflared runs on this same machine and connects
to it locally, so the proxy itself never needs to listen on a public interface.
"""

from __future__ import annotations

import os
from secrets import compare_digest

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.getenv("PROXY_UPSTREAM", "http://localhost:1234").rstrip("/")

# Fail at startup rather than silently booting an open endpoint. An empty or
# missing token is the one configuration mistake that would defeat the point of
# this file, so it is not allowed to have a default.
TOKEN = os.getenv("PROXY_TOKEN", "")
if not TOKEN:
    raise SystemExit(
        "PROXY_TOKEN is not set. Refusing to start: an unauthenticated proxy "
        "would expose LM Studio to anyone who finds the tunnel URL."
    )

# Long, because a cold model load plus a 1024-token completion on a 4B model can
# legitimately take a while; the app's own client gives up at 120 s.
TIMEOUT = httpx.Timeout(180.0, connect=10.0)

# Headers that describe a single hop and must not be replayed onto the next one.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def _authorized(request: Request) -> bool:
    """True when the request carries the expected bearer token.

    compare_digest rather than `==` so that a timing signal cannot be used to
    recover the token byte by byte.
    """
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return compare_digest(presented.strip(), TOKEN)


async def proxy(request: Request) -> StreamingResponse | JSONResponse:
    if not _authorized(request):
        # Deliberately says nothing about whether the path or the model exists.
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    url = f"{UPSTREAM}/{request.path_params['path'].lstrip('/')}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}

    client = httpx.AsyncClient(timeout=TIMEOUT)
    upstream_request = client.build_request(
        request.method,
        url,
        headers=headers,
        params=request.query_params,
        content=await request.body(),
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        return JSONResponse(
            {"error": f"LM Studio is not reachable at {UPSTREAM}. Is the server running?"},
            status_code=502,
        )
    except httpx.TimeoutException:
        await client.aclose()
        return JSONResponse({"error": "upstream timed out"}, status_code=504)

    # Streamed rather than buffered so that a token-by-token SSE response is
    # relayed as it arrives instead of being held until the completion ends.
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers={
            k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
        },
        background=BackgroundTask(_close, upstream, client),
    )


async def _close(upstream: httpx.Response, client: httpx.AsyncClient) -> None:
    await upstream.aclose()
    await client.aclose()


async def health(_: Request) -> JSONResponse:
    """Unauthenticated liveness check, so you can confirm the tunnel is up
    without putting the token in a browser URL bar. Reveals nothing."""
    return JSONResponse({"status": "ok"})


app = Starlette(
    routes=[
        Route("/healthz", health, methods=["GET"]),
        Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
    ]
)
