"""Send ONE request to the API in your .env and print the full answer or the full error (never the key).
Usage:  python diagnose_api.py            (reads .env)
        python diagnose_api.py .env.other
"""
import re, sys
from openai import OpenAI
path = sys.argv[1] if len(sys.argv) > 1 else ".env"
t = open(path, encoding="utf-8").read()
def get(name):
    m = re.search(rf"^{name}=(.*)$", t, re.M)
    return m.group(1).strip().strip('"').strip("'") if m else None
key, base, models = get("OPENAI_API_KEY"), get("OPENAI_BASE_URL"), get("MODELS")
model = models.split(":", 1)[1] if models and ":" in models else models
print("base url :", base)
print("model    :", model)
print("key      :", ("present, %d characters, starts with %r" % (len(key), key[:4])) if key else "MISSING")
c = OpenAI(base_url=base, api_key=key or "EMPTY", max_retries=0)
try:
    r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": "Reply with the single word OK."}], max_tokens=20)
    print("\nSUCCESS. The model answered:", repr(r.choices[0].message.content))
except Exception as e:
    print("\nFAILED:", type(e).__name__, "| status:", getattr(e, "status_code", None))
    print("message :", getattr(e, "message", str(e))[:600])
    print("body    :", str(getattr(e, "body", None))[:600])
    h = getattr(getattr(e, "response", None), "headers", {}) or {}
    for k, v in h.items():
        if any(w in k.lower() for w in ("rate", "retry", "limit", "remaining", "reset")):
            print("header  :", k, "=", v)