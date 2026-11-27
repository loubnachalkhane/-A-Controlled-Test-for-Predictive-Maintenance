"""Print each saved answer with its stop reason and length. Usage: python show_answers.py agent_runs_groq_test"""
import glob, json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "agent_runs_groq_test"
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    recs = [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [r for r in recs if r.get("response")]
    if not recs:
        continue
    print("=" * 70, "\n", os.path.basename(f), f"({len(recs)} answers)")
    lens = [len(r["response"]) for r in recs]
    print(f"answer length in characters: min {min(lens)}, mean {sum(lens)/len(lens):.0f}, max {max(lens)}")
    print("contains '<think>' tag:", any("<think>" in r["response"] for r in recs))
    for r in recs[:3]:
        print(f"\n--- window {r['idx']}  F={r['F']:.4f}  decision={r['decision']}  stop={r.get('stop_reason')}  chars={len(r['response'])}")
        print(r["response"])