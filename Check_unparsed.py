"""Show why some answers were unparseable. Usage: python check_unparsed.py agent_runs"""
import glob, json, os, sys
from collections import Counter
d = sys.argv[1] if len(sys.argv) > 1 else "agent_runs"
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    bad = [r for r in recs if r["decision"] is None]
    print(f"\n{os.path.basename(f)}: {len(bad)} unparseable of {len(recs)}")
    if not bad:
        continue
    print("   stop reasons:", dict(Counter(r.get("stop_reason", "unknown (old file)") for r in bad)))
    lens = [len(r["response"] or "") for r in bad]
    print("   answer length (characters) of the unparseable ones: min %d, max %d" % (min(lens), max(lens)))
    print("   mean F of the unparseable windows: %.3f   (tau is near 0.1)" % (sum(r["F"] for r in bad) / len(bad)))
    for r in bad[:2]:
        t = (r["response"] or "")
        print("   ---- window", r["idx"], "F=%.4f" % r["F"], "| ends with: ...", repr(t[-160:]))