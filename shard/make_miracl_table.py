"""Format exp17b_miracl.json into LaTeX rows matching tab:shard-beir style."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS

J = RESULTS / "exp17b_outputs" / "exp17b_miracl.json"
res = json.load(open(J, encoding="utf-8"))["results"]
LANG = {"sw": "Swahili", "bn": "Bengali", "te": "Telugu", "ru": "Russian"}
ENC = {"intfloat/multilingual-e5-base": "e5-base", "intfloat/multilingual-e5-small": "e5-small"}


def b(x):                                   # compact CI bound: -0.073 -> {-}.073
    return ("{-}" if x < 0 else "") + f"{abs(x):.3f}"[1:]


def ci(o, bold=False):
    s = f"{o['delta']:+.3f}\\,[{b(o['lo'])},{b(o['hi'])}]"
    return ("\\mathbf{" + s + "}") if bold else s


# order rows: e5-base then e5-small, sw then bn
def keyf(r):
    return (0 if "base" in r["encoder"] else 1, list(LANG).index(r["lang"]))


print("% --- tab:shard-miracl rows ---")
for r in sorted(res, key=keyf):
    d = r["d"]; d4 = f"dpub{round(d/4)}_kc200"; d8 = f"dpub{round(d/8)}_kc200"
    enc = ENC[r["encoder"]]; lang = LANG[r["lang"]]; n = round(r["n_corpus"] / 1000)
    raw = r["raw_ndcg10"]
    svd = ci(r["svd_vs_raw"])
    sh4 = ci(r["shard"][d4]["vs_raw"], bold=True)
    sh8 = ci(r["shard"][d8]["vs_raw"])
    print(f"{enc:8s} & {lang} (${n}$k) & ${raw:.3f}$ & ${svd}$ & ${sh4}$ & ${sh8}$\\\\")

print("\n% --- summary numbers ---")
svd_losses = [r["svd_vs_raw"]["delta"] for r in res]
print(f"SVD loss range: {min(svd_losses):+.3f} .. {max(svd_losses):+.3f}")
for r in sorted(res, key=keyf):
    d = r["d"]; d4 = f"dpub{round(d/4)}_kc200"
    o = r["shard"][d4]["vs_raw"]
    print(f"  {ENC[r['encoder']]:8s} {r['lang']}: raw={r['raw_ndcg10']:.3f} "
          f"svd_d2={r['svd_ndcg10']:.3f} shard_d4 delta={o['delta']:+.3f} [{o['lo']:+.3f},{o['hi']:+.3f}] "
          f"nq={r['n_q']} |C|={r['n_corpus']}")
