"""Memory-capped adaptive/few-shot inversion experiment.

This is the portable "heavy" inversion experiment referenced by the paper.
It used to be sized for an A100; it is now designed to run on an RTX 4090
with roughly 11 GB of free VRAM by default, while retaining an A100 profile.

Attack being evaluated:

1. encode texts with the GTR-base embedder used by Vec2Text;
2. protect embeddings with SVD truncation and a secret orthogonal rotation;
3. reveal m known plaintext pairs to the attacker;
4. recover the projected native space with orthogonal Procrustes;
5. lift recovered vectors back to native embedding dimension and run the
   official Vec2Text corrector with a stronger-than-paper decoding budget.

The RTX4090 profile is intentionally slower and more conservative than the
A100 profile: small inversion batches, short max input length, checkpointed
per-case output, and automatic retry with smaller batches on CUDA OOM.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paths import DATA, RESULTS, FIGS
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

np: Any
torch: Any
load_dataset: Any
randomized_svd: Any
tqdm: Any
vec2text: Any


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "rtx4090_11gb": {
        "max_texts": 2200,
        "n_test": 120,
        "known_pairs": "0,10,50,100,192,500",
        "encode_batch_size": 24,
        "invert_batch_size": 2,
        "vec2text_steps": 24,
        "beam_width": 1,
        "max_length": 96,
        "torch_dtype": "float16",
    },
    "rtx4090_11gb_smoke": {
        "max_texts": 900,
        "n_test": 24,
        "known_pairs": "0,50,192",
        "encode_batch_size": 16,
        "invert_batch_size": 1,
        "vec2text_steps": 8,
        "beam_width": 1,
        "max_length": 64,
        "torch_dtype": "float16",
    },
    "a100": {
        "max_texts": 6000,
        "n_test": 500,
        "known_pairs": "0,10,50,100,250,500,1000",
        "encode_batch_size": 128,
        "invert_batch_size": 32,
        "vec2text_steps": 50,
        "beam_width": 4,
        "max_length": 128,
        "torch_dtype": "bfloat16",
    },
}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_ssl_verification(disable: bool) -> None:
    if not disable:
        return

    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    os.environ["PYTHONHTTPSVERIFY"] = "0"

    try:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception as exc:
        print(f"warning: could not patch Python SSL context: {exc}")

    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception as exc:
        print(f"warning: could not disable urllib3 SSL warnings: {exc}")

    try:
        import requests

        if not getattr(requests.sessions.Session.request, "_ssl_verify_disabled", False):
            original_request = requests.sessions.Session.request

            def request_without_ssl_verify(self, method, url, **kwargs):
                kwargs["verify"] = False
                return original_request(self, method, url, **kwargs)

            request_without_ssl_verify._ssl_verify_disabled = True
            requests.sessions.Session.request = request_without_ssl_verify

        try:
            from huggingface_hub import configure_http_backend

            def backend_factory() -> requests.Session:
                session = requests.Session()
                session.verify = False
                return session

            configure_http_backend(backend_factory=backend_factory)
        except Exception as exc:
            print(f"warning: could not configure Hugging Face HTTP backend: {exc}")
    except Exception as exc:
        print(f"warning: could not patch requests SSL verification: {exc}")

    print("warning: TLS certificate verification is disabled for this process")


def import_runtime_dependencies() -> None:
    global np, torch, load_dataset, randomized_svd, tqdm, vec2text

    import numpy as _np
    import torch as _torch
    from datasets import load_dataset as _load_dataset
    from sklearn.utils.extmath import randomized_svd as _randomized_svd
    from tqdm.auto import tqdm as _tqdm

    import vec2text as _vec2text

    np = _np
    torch = _torch
    load_dataset = _load_dataset
    randomized_svd = _randomized_svd
    tqdm = _tqdm
    vec2text = _vec2text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="rtx4090_11gb")
    p.add_argument("--output-dir", type=Path, default=Path("results/adaptive_inversion_outputs"))
    p.add_argument("--dataset", type=str, default="ag_news")
    p.add_argument("--dataset-split", type=str, default="test")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--texts-jsonl", type=Path, default=None)
    p.add_argument(
        "--synthetic-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a built-in synthetic news corpus if the Hugging Face dataset cannot be loaded.",
    )
    p.add_argument(
        "--disable-ssl-verification",
        action=argparse.BooleanOptionalAction,
        default=env_flag("HF_DISABLE_SSL_VERIFY"),
        help="Disable TLS certificate verification for Hugging Face/requests downloads.",
    )
    p.add_argument("--max-texts", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--k-frac", type=float, default=0.5)
    p.add_argument("--known-pairs", type=str, default=None)
    p.add_argument("--rotation-seed", type=int, default=11)
    p.add_argument("--sample-seed", type=int, default=20260530)
    p.add_argument("--encode-batch-size", type=int, default=None)
    p.add_argument("--invert-batch-size", type=int, default=None)
    p.add_argument("--min-invert-batch-size", type=int, default=1)
    p.add_argument("--vec2text-steps", type=int, default=None)
    p.add_argument("--beam-width", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default=None)
    p.add_argument("--skip-raw", action="store_true", help="Skip raw-embedding inversion to save time.")
    p.add_argument("--skip-known-r", action="store_true", help="Skip known-R oracle inversion.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-every-case", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only rebuild summary JSON/CSV from existing case_*.json files in --output-dir.",
    )
    args = p.parse_args()
    apply_profile_defaults(args)
    return args


def apply_profile_defaults(args: argparse.Namespace) -> None:
    defaults = PROFILE_DEFAULTS[args.profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def dtype_from_arg(arg: str, device: str) -> torch.dtype:
    if arg == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[arg]


def log_gpu(prefix: str, device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        print(f"{prefix}: CUDA unavailable or disabled")
        return
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(idx)
    name = torch.cuda.get_device_name(idx)
    print(
        f"{prefix}: {name}, free={free / 2**30:.2f} GiB, "
        f"total={total / 2**30:.2f} GiB"
    )


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def move_corrector_to_device(corrector, device: str, dtype: torch.dtype) -> None:
    """Move known torch modules inside vec2text's corrector object.

    The vec2text package has changed object internals across releases. We avoid
    relying on a single attribute layout and move every directly attached
    torch.nn.Module we can see. Float32 embeddings are still passed to
    vec2text.invert_embeddings; this only reduces model activation/weight VRAM.
    """

    for name, value in vars(corrector).items():
        if isinstance(value, torch.nn.Module):
            try:
                value.to(device)
                if dtype in (torch.float16, torch.bfloat16):
                    value.to(dtype=dtype)
                print(f"moved corrector.{name} to {device} ({dtype})")
            except Exception as exc:
                print(f"warning: could not move corrector.{name}: {exc}")


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.texts_jsonl is not None:
        texts = []
        with args.texts_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(str(obj.get(args.text_field, "")).strip())
                if len(texts) >= args.max_texts:
                    break
        return [t for t in texts if t]

    if args.dataset.lower() in {"synthetic", "synthetic_news", "offline"}:
        return make_synthetic_news(args.max_texts, args.sample_seed)

    try:
        ds = load_dataset(args.dataset, split=args.dataset_split)
    except Exception as exc:
        if not args.synthetic_fallback:
            raise RuntimeError(
                f"Could not load Hugging Face dataset {args.dataset!r}. "
                "Use --dataset synthetic_news, --texts-jsonl, or fix the Hub SSL/proxy setup."
            ) from exc
        print(
            f"warning: could not load dataset {args.dataset!r} "
            f"({exc.__class__.__name__}: {exc}); using built-in synthetic_news fallback"
        )
        return make_synthetic_news(args.max_texts, args.sample_seed)

    n = min(args.max_texts, len(ds))
    texts = [str(x[args.text_field]).strip() for x in ds.select(range(n))]
    return [t for t in texts if t]


def make_synthetic_news(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    regions = [
        "Boston",
        "Denver",
        "Seattle",
        "Austin",
        "Chicago",
        "Madrid",
        "Berlin",
        "Toronto",
        "Singapore",
        "Dublin",
    ]
    topics = [
        "energy storage",
        "clinical diagnostics",
        "satellite imaging",
        "public transit",
        "crop monitoring",
        "cybersecurity",
        "language technology",
        "supply chains",
        "urban planning",
        "financial compliance",
    ]
    organizations = [
        "university researchers",
        "city officials",
        "startup engineers",
        "hospital administrators",
        "regulatory analysts",
        "industry consortium members",
        "public health teams",
        "logistics operators",
    ]
    actions = [
        "reported a pilot deployment",
        "released a benchmark study",
        "approved a procurement plan",
        "found a measurable efficiency gain",
        "identified a reliability problem",
        "announced a multi-year evaluation",
        "published an audit summary",
        "expanded a field trial",
    ]
    details = [
        "The decision follows six months of testing under realistic load.",
        "Independent reviewers said the evidence is promising but incomplete.",
        "The team will compare the results against a conventional baseline.",
        "Analysts highlighted cost, latency and privacy as the main trade-offs.",
        "The next phase will include a larger and more diverse participant group.",
        "Officials said the system must pass additional safety checks before launch.",
    ]

    out = []
    for i in range(n):
        region = regions[int(rng.integers(len(regions)))]
        topic = topics[int(rng.integers(len(topics)))]
        org = organizations[int(rng.integers(len(organizations)))]
        action = actions[int(rng.integers(len(actions)))]
        detail = details[int(rng.integers(len(details)))]
        value = int(rng.integers(12, 95))
        quarter = int(rng.integers(1, 5))
        out.append(
            f"In {region}, {org} {action} for {topic}. "
            f"The project recorded a {value} percent change in its primary metric during quarter {quarter}. "
            f"{detail}"
        )
    return out


def make_synthetic_pii(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    names = ["John Smith", "Maria Garcia", "Ivan Petrov", "Emily Chen", "Fatima Khan"]
    cities = ["Boston", "Denver", "Eugene", "Austin", "Seattle"]
    streets = ["Oak Street", "Pine Road", "Lake Avenue", "Hill Lane", "Cedar Drive"]
    out = []
    for _ in range(n):
        name = names[int(rng.integers(len(names)))]
        city = cities[int(rng.integers(len(cities)))]
        street = streets[int(rng.integers(len(streets)))]
        phone = f"({int(rng.integers(200,999))}) {int(rng.integers(100,999))}-{int(rng.integers(1000,9999))}"
        card = " ".join(f"{int(rng.integers(0,9999)):04d}" for _ in range(4))
        out.append(
            f"{name} lives at {int(rng.integers(10,999))} {street} in {city}. "
            f"His phone number is {phone}. Payment card {card} was used at the clinic."
        )
    return out


def encode_batch(model, tok, batch: list[str], args: argparse.Namespace) -> np.ndarray:
    with torch.inference_mode():
        inp = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        out = model(**inp)
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        return emb.detach().cpu().numpy().astype(np.float32)


def encode_with_corrector(corrector, texts: list[str], args: argparse.Namespace) -> np.ndarray:
    model = corrector.embedder.to(args.device)
    tok = corrector.embedder_tokenizer
    batch_size = args.encode_batch_size
    vecs = []
    i = 0
    pbar = tqdm(total=len(texts), desc="encoding")
    while i < len(texts):
        batch = texts[i : i + batch_size]
        try:
            vecs.append(encode_batch(model, tok, batch, args))
            i += len(batch)
            pbar.update(len(batch))
        except torch.cuda.OutOfMemoryError:
            clear_cuda()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"CUDA OOM during encoding; retrying with encode_batch_size={batch_size}")
    pbar.close()
    return np.vstack(vecs).astype(np.float32)


def make_random_orthogonal(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim), dtype=np.float32)
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    c = source.T @ target
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    return (u @ vt).astype(np.float32)


def reconstruct_native(y: np.ndarray, w: np.ndarray, vk: np.ndarray, mu: np.ndarray) -> np.ndarray:
    z_hat = y @ w
    x_hat = z_hat @ vk.T + mu
    x_hat /= np.maximum(np.linalg.norm(x_hat, axis=1, keepdims=True), 1e-12)
    return x_hat.astype(np.float32)


def accepts_keyword(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def corrector_embedding_dtype(corrector) -> torch.dtype:
    for attr in ("model", "corrector", "embedder"):
        module = getattr(corrector, attr, None)
        if isinstance(module, torch.nn.Module):
            for param in module.parameters():
                if param.is_floating_point():
                    return param.dtype
    return torch.float32


def call_vec2text(corrector, batch: np.ndarray, batch_size: int, args: argparse.Namespace) -> list[str]:
    emb_dtype = corrector_embedding_dtype(corrector)
    emb = torch.tensor(batch, dtype=emb_dtype, device=args.device)
    kwargs: dict[str, Any] = {
        "embeddings": emb,
        "corrector": corrector,
        "num_steps": args.vec2text_steps,
    }
    if accepts_keyword(vec2text.invert_embeddings, "sequence_beam_width"):
        kwargs["sequence_beam_width"] = args.beam_width
    elif accepts_keyword(vec2text.invert_embeddings, "beam_width"):
        kwargs["beam_width"] = args.beam_width
    if accepts_keyword(vec2text.invert_embeddings, "batch_size"):
        kwargs["batch_size"] = batch_size

    with torch.inference_mode():
        recovered = vec2text.invert_embeddings(**kwargs)
    del emb
    clear_cuda()
    return recovered


def invert_embeddings(corrector, emb: np.ndarray, case: str, args: argparse.Namespace) -> list[str]:
    outs: list[str] = []
    batch_size = args.invert_batch_size
    i = 0
    pbar = tqdm(total=len(emb), desc=f"vec2text:{case}")
    while i < len(emb):
        batch = emb[i : i + batch_size]
        try:
            recovered = call_vec2text(corrector, batch, len(batch), args)
            outs.extend(recovered)
            i += len(batch)
            pbar.update(len(batch))
        except torch.cuda.OutOfMemoryError:
            clear_cuda()
            if batch_size <= args.min_invert_batch_size:
                raise
            batch_size = max(args.min_invert_batch_size, batch_size // 2)
            print(f"CUDA OOM during {case}; retrying with invert_batch_size={batch_size}")
    pbar.close()
    return outs


def tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def token_f1(ref: str, hyp: str) -> float:
    r = tokenize(ref)
    h = tokenize(hyp)
    if not r or not h:
        return 0.0
    r_counts = {}
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1
    overlap = 0
    for t in h:
        if r_counts.get(t, 0) > 0:
            overlap += 1
            r_counts[t] -= 1
    prec = overlap / len(h)
    rec = overlap / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


PII_PATTERNS = {
    "phone": re.compile(r"\(?\d{3}\)?[- ]\d{3}[- ]\d{4}"),
    "card": re.compile(r"(?:\d{4}[ -]){3}\d{4}"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}


def pii_recall(ref: str, hyp: str) -> float:
    ref_items = []
    hyp_items = []
    for pat in PII_PATTERNS.values():
        ref_items.extend(pat.findall(ref))
        hyp_items.extend(pat.findall(hyp))
    if not ref_items:
        return float("nan")
    hyp_norm = {"".join(re.findall(r"\w", x.lower())) for x in hyp_items}
    hits = 0
    for item in ref_items:
        if "".join(re.findall(r"\w", item.lower())) in hyp_norm:
            hits += 1
    return hits / len(ref_items)


def evaluate_case(case: str, refs: list[str], hyps: list[str]) -> dict[str, float | str]:
    token_scores = np.array([token_f1(r, h) for r, h in zip(refs, hyps)], dtype=np.float64)
    pii_scores = np.array([pii_recall(r, h) for r, h in zip(refs, hyps)], dtype=np.float64)
    rec: dict[str, float | str] = {
        "case": case,
        "token_f1_mean": float(token_scores.mean()),
        "exact_match_rate": float(np.mean([r.strip() == h.strip() for r, h in zip(refs, hyps)])),
        "pii_recall_mean": float(np.nanmean(pii_scores)) if np.any(~np.isnan(pii_scores)) else float("nan"),
    }
    try:
        import sacrebleu

        bleu = [sacrebleu.sentence_bleu(h, [r]).score / 100.0 for r, h in zip(refs, hyps)]
        rec["bleu_mean"] = float(np.mean(bleu))
    except Exception:
        rec["bleu_mean"] = float("nan")
    return rec


def safe_case_name(case: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case)


def load_completed_cases(output_dir: Path) -> set[str]:
    return {p.stem.removeprefix("case_") for p in output_dir.glob("case_*.json")}


def save_case(output_dir: Path, case: str, summary: dict[str, Any], refs: list[str], hyps: list[str]) -> None:
    safe = safe_case_name(case)
    payload = {
        "summary": summary,
        "samples": [
            {"sample_id": i, "reference": ref, "recovered": hyp}
            for i, (ref, hyp) in enumerate(zip(refs, hyps))
        ],
    }
    (output_dir / f"case_{safe}.json").write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def build_cases(
    args: argparse.Namespace,
    x: np.ndarray,
    texts: list[str],
) -> tuple[dict[str, np.ndarray], list[str], int, int]:
    n, d = x.shape
    k = int(round(d * args.k_frac))
    mu = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mu
    _, _, vt = randomized_svd(xc, n_components=k, random_state=args.sample_seed, n_iter=5)
    vk = vt.T.astype(np.float32)
    z = (xc @ vk).astype(np.float32)
    r = make_random_orthogonal(args.rotation_seed, k)
    y = z @ r

    known_sizes = [int(s) for s in args.known_pairs.split(",") if s.strip()]
    n_anchor_max = max(known_sizes)
    if n_anchor_max + args.n_test > n:
        raise ValueError(
            f"Need more texts: max known pairs {n_anchor_max} + n_test {args.n_test} > n={n}"
        )

    anchor_z = z[:n_anchor_max]
    anchor_y = y[:n_anchor_max]
    test_y = y[n_anchor_max : n_anchor_max + args.n_test]
    test_refs = texts[n_anchor_max : n_anchor_max + args.n_test]
    test_x = x[n_anchor_max : n_anchor_max + args.n_test]

    cases: dict[str, np.ndarray] = {}
    if not args.skip_raw:
        cases["raw"] = test_x
    cases["unknown_rotation_no_alignment"] = reconstruct_native(test_y, np.eye(k, dtype=np.float32), vk, mu)
    if not args.skip_known_r:
        cases["known_R_oracle"] = reconstruct_native(test_y, r.T, vk, mu)
    for m in known_sizes:
        if m == 0:
            w = np.eye(k, dtype=np.float32)
        else:
            w = procrustes(anchor_y[:m], anchor_z[:m])
        cases[f"fewshot_procrustes_m{m}"] = reconstruct_native(test_y, w, vk, mu)
    return cases, test_refs, d, k


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_final_outputs(
    output_dir: Path,
    config: dict[str, Any],
    n: int | None,
    d: int | None,
    k: int | None,
) -> None:
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata.get("config", config)
        n = metadata.get("n", n)
        d = metadata.get("d", d)
        k = metadata.get("k", k)

    summaries = []
    sample_rows = []
    case_files = sorted(output_dir.glob("case_*.json"))
    if not case_files:
        raise RuntimeError(f"No case_*.json files found in {output_dir}")

    for p in case_files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        summaries.append(payload["summary"])
        case = payload["summary"]["case"]
        for row in payload["samples"]:
            sample_rows.append(
                {
                    "case": case,
                    "sample_id": row["sample_id"],
                    "reference": row["reference"],
                    "recovered": row["recovered"],
                }
            )

    (output_dir / "adaptive_inversion_summary.json").write_text(
        json.dumps(
            {"config": config, "n": n, "d": d, "k": k, "summary": summaries},
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    with (output_dir / "adaptive_inversion_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "sample_id", "reference", "recovered"])
        writer.writeheader()
        writer.writerows(sample_rows)


def save_run_metadata(output_dir: Path, config: dict[str, Any], n: int, d: int, k: int) -> None:
    (output_dir / "run_metadata.json").write_text(
        json.dumps({"config": config, "n": n, "d": d, "k": k}, indent=2, default=json_default),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.finalize_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_final_outputs(args.output_dir, vars(args), None, None, None)
        print(f"finalized outputs in {args.output_dir}")
        return

    configure_ssl_verification(args.disable_ssl_verification)
    import_runtime_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}. "
            "On RTX 4090 hosts with NVIDIA driver 550.x, recreate the virtual "
            "environment and install docs/requirements_rtx4090.txt so pip "
            "uses the pinned CUDA 12.4 PyTorch wheel instead of a CUDA 13 wheel."
        )

    dtype = dtype_from_arg(args.torch_dtype, args.device)
    log_gpu("startup", args.device)
    print(
        f"profile={args.profile}, max_texts={args.max_texts}, n_test={args.n_test}, "
        f"known_pairs={args.known_pairs}, invert_batch={args.invert_batch_size}, "
        f"steps={args.vec2text_steps}, beam={args.beam_width}, max_length={args.max_length}, dtype={dtype}"
    )

    texts = load_texts(args)
    texts += make_synthetic_pii(max(200, args.n_test // 2), args.sample_seed)
    rng = np.random.default_rng(args.sample_seed)
    rng.shuffle(texts)
    texts = texts[: args.max_texts]

    corrector = vec2text.load_pretrained_corrector("gtr-base")
    move_corrector_to_device(corrector, args.device, dtype)
    log_gpu("after model load", args.device)

    x = encode_with_corrector(corrector, texts, args)
    cases, test_refs, d, k = build_cases(args, x, texts)
    save_run_metadata(args.output_dir, vars(args), len(texts), d, k)
    completed = load_completed_cases(args.output_dir) if args.resume else set()

    for case, emb in cases.items():
        safe = safe_case_name(case)
        if safe in completed:
            print(f"skipping completed case {case}")
            continue
        t0 = time.time()
        hyps = invert_embeddings(corrector, emb, case, args)
        elapsed = time.time() - t0
        summary = evaluate_case(case, test_refs, hyps)
        summary["seconds"] = elapsed
        if args.save_every_case:
            save_case(args.output_dir, case, summary, test_refs, hyps)
        print(json.dumps(summary, indent=2))

    write_final_outputs(args.output_dir, vars(args), len(texts), d, k)
    log_gpu("done", args.device)


if __name__ == "__main__":
    main()
