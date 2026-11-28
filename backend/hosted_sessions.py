"""Short-lived server-side sessions shared across Vercel instances."""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException


def _credentials():
    return (
        os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("KV_REST_API_URL"),
        os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("KV_REST_API_TOKEN"),
    )


def enabled():
    return os.getenv("VERCEL") == "1" or bool(_credentials()[0])


def _command(*parts):
    url, token = _credentials()
    if not url or not token or not url.startswith("https://"):
        raise HTTPException(503, "Shared session storage is not configured.")
    request = Request(url, data=json.dumps(parts).encode(), headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=5) as response:
            data = json.load(response)
        if "error" in data or "result" not in data:
            raise ValueError("Unexpected storage response")
        return data["result"]
    except (OSError, URLError, ValueError) as exc:
        raise HTTPException(503, "Shared session storage is temporarily unavailable.") from exc


def save(namespace, session_id, value):
    _command("SET", _key(namespace, session_id), value, "EX", 1800)


def read(namespace, session_id):
    return _command("GET", _key(namespace, session_id))


def _key(namespace, session_id):
    deployment = os.getenv("VERCEL_URL", "local")
    return f"qkd:{deployment}:{namespace}:{session_id}"
