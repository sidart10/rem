"""Minimal fal.ai queue client (standard library only).

Needs FAL_KEY in the environment. Get one at https://fal.ai/dashboard/keys, then `export FAL_KEY=...` (or put it in
your shell profile). The key is read here and sent only to fal.
"""
import base64
import json
import mimetypes
import os
import time
import urllib.request


def _key():
    k = os.environ.get("FAL_KEY")
    if not k:
        raise SystemExit("FAL_KEY is not set. Get a key at https://fal.ai/dashboard/keys and `export FAL_KEY=...`. "
                         "Only SAM 3 masking needs it; alpha masks and every measuring step run locally.")
    return k


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def _req(url, payload=None, method="GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Authorization": f"Key {_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())


def run(endpoint, arguments, timeout=600):
    """Submit to the fal queue, wait, return the result JSON."""
    sub = _req(f"https://queue.fal.run/{endpoint}", arguments, "POST")
    status_url, response_url = sub["status_url"], sub["response_url"]
    t0 = time.time()
    while True:
        st = _req(status_url)
        if st.get("status") == "COMPLETED":
            if st.get("error"):  # fal reports a failed request as COMPLETED with error fields
                raise RuntimeError(f"fal {endpoint} failed: {st.get('error_type')}: {st['error']}")
            return _req(response_url)
        if st.get("status") in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal {endpoint} failed: {st}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"fal {endpoint} still running after {timeout}s; request {sub.get('request_id')}")
        time.sleep(2)


def download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst
