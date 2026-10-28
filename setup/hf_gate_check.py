"""Which Hugging Face repos used by the study are gated, and can we read them anonymously?

Uses the public HF API only (no token is sent), so the answer is what a
fresh machine without the author's login would see.
"""
import json
import urllib.error
import urllib.request

REPOS = [
    "Prior-Labs/tabpfn_3_5", "Prior-Labs/tabpfn_3", "Prior-Labs/tabpfn_2_6",
    "Prior-Labs/tabpfn_2_5", "Prior-Labs/TabPFN-v2-clf",
    "autogluon/mitra-classifier-2", "autogluon/mitra-classifier",
    "jingang/TabICL-clf", "jingang/TabICL",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "env-check"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


for repo in REPOS:
    try:
        info = get(f"https://huggingface.co/api/models/{repo}")
        files = [s["rfilename"] for s in info.get("siblings", [])]
        print(f"{repo}: gated={info.get('gated')} private={info.get('private')} "
              f"lastModified={info.get('lastModified')} sha={info.get('sha', '')[:12]}")
        print("    files:", ", ".join(f for f in files if not f.startswith(".")))
    except urllib.error.HTTPError as e:
        print(f"{repo}: HTTP {e.code}")
    except Exception as e:
        print(f"{repo}: ERROR {e}")
