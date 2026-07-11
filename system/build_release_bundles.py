"""Build clean journal, arXiv, and reproducibility delivery bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Iterable
from zipfile import ZipInfo


ARXIV_SOURCE_FILES = (
    "main.tex",
    "main.bbl",
    "revised_refs.bib",
    "sn-jnl.cls",
    "sn-basic.bst",
    "fig_kernel_scaling.pdf",
    "fig_system_scaling.pdf",
    "fig_svd_pareto.pdf",
    "fig_tradeoff.pdf",
    "fig_leakage.pdf",
    "fig_beir_utility.pdf",
)

PUBLIC_DOCS = (
    "ONLINE_RESOURCE_1_README.txt",
    "THIRD_PARTY_NOTICES.md",
    "environment.lock",
    "reproduce_system_revision.md",
    "results_manifest.md",
    "ir_expansion.md",
    "security_expansion.md",
    "systems_expansion.md",
    "confirmatory_analysis_plan.md",
    "arxiv_revision_delta.md",
)

ROOT_ARTIFACT_FILES = (
    ".gitignore",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "Makefile",
    "requirements.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_reset(path: Path, output_root: Path) -> None:
    path = path.resolve()
    output_root = output_root.resolve()
    if path == output_root or output_root not in path.parents:
        raise ValueError(f"refusing to reset path outside a delivery subdirectory: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_files(source_dir: Path, destination: Path, names: Iterable[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / source.name)


def copy_tree(
    source: Path,
    destination: Path,
    include: Callable[[Path], bool],
) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if include(relative):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def write_checksums(root: Path) -> None:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "manifest.json"}:
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "packrerank_release_manifest.v1",
                "file_count": len(records),
                "total_bytes": sum(item["bytes"] for item in records),
                "files": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )


def write_zip(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                info = ZipInfo(
                    path.relative_to(source).as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes(), compresslevel=9)


def include_system(relative: Path) -> bool:
    lowered = {part.lower() for part in relative.parts}
    if "__pycache__" in lowered or relative.suffix.lower() in {".pyc", ".log"}:
        return False
    if relative.name in {"localhost_key.pem", "localhost_cert.pem"}:
        return False
    return True


def include_result(relative: Path) -> bool:
    posix = relative.as_posix()
    if posix == "cache/E_docs_e5base_proj672_1000000.npy":
        return False
    if relative.name == "systems_expansion_smoke.json":
        return False
    if relative.name.endswith(".sage.py") or relative.suffix.lower() == ".log":
        return False
    if relative.name in {
        "wsl_apt_install_stdout.txt",
        "wsl_install_stdout.txt",
        "wsl_ubuntu_apt_install_stdout.txt",
        "wsl_ubuntu_install_stdout.txt",
    }:
        return False
    return True


def build(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    submission = output / "submission"
    reproducibility = output / "reproducibility"
    pdf_output = output / "pdf"
    for directory in (submission, reproducibility, pdf_output):
        safe_reset(directory, output)

    arxiv_source = submission / "kurilenko_arxiv_source"
    journal_source = submission / "kurilenko_jsc_source"
    arxiv_metadata = submission / "arxiv_metadata"
    for directory in (arxiv_source, journal_source, arxiv_metadata):
        directory.mkdir(parents=True, exist_ok=True)

    paper = root / "paper_revised"
    copy_files(paper, arxiv_source, ARXIV_SOURCE_FILES)
    shutil.copy2(root / "submission" / "README_ARXIV.txt", arxiv_source / "README.txt")
    copy_files(paper, journal_source, ARXIV_SOURCE_FILES)
    shutil.copy2(paper / "README_SUBMISSION.txt", journal_source / "README.txt")

    copy_files(
        root / "submission",
        arxiv_metadata,
        ("arxiv_abstract.txt", "arxiv_metadata.md", "arxiv_replacement_note.txt"),
    )
    write_checksums(arxiv_source)
    write_checksums(journal_source)
    write_checksums(arxiv_metadata)

    artifact = reproducibility / "kurilenko_online_resource_1"
    artifact.mkdir(parents=True, exist_ok=True)
    copy_files(root, artifact, ROOT_ARTIFACT_FILES)
    shutil.copy2(root / "docs" / "ONLINE_RESOURCE_1_README.txt", artifact / "README.txt")
    copy_files(root / "docs", artifact / "docs", PUBLIC_DOCS[1:])
    copy_files(paper, artifact / "manuscript", ARXIV_SOURCE_FILES)
    shutil.copy2(paper / "README_SUBMISSION.txt", artifact / "manuscript" / "README.txt")
    copy_tree(root / "system", artifact / "system", include_system)
    copy_tree(root / "tests", artifact / "tests", lambda p: p.suffix == ".py")
    copy_tree(
        root / "results" / "system_revision",
        artifact / "results" / "system_revision",
        include_result,
    )
    write_checksums(artifact)

    arxiv_zip = submission / "kurilenko_arxiv_source.zip"
    journal_zip = submission / "kurilenko_jsc_source.zip"
    artifact_zip = reproducibility / "kurilenko_online_resource_1.zip"
    write_zip(arxiv_source, arxiv_zip)
    write_zip(journal_source, journal_zip)
    write_zip(artifact, artifact_zip)

    shutil.copy2(paper / "main.pdf", pdf_output / "kurilenko_jsc_revised.pdf")
    cover_letter = root / "submission" / "cover_letter.pdf"
    if cover_letter.is_file():
        shutil.copy2(cover_letter, pdf_output / "cover_letter.pdf")

    deliverables = [
        arxiv_zip,
        journal_zip,
        artifact_zip,
        pdf_output / "kurilenko_jsc_revised.pdf",
    ]
    if (pdf_output / "cover_letter.pdf").is_file():
        deliverables.append(pdf_output / "cover_letter.pdf")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in deliverables
        ),
        encoding="utf-8",
    )
    return {
        "schema": "packrerank_delivery.v1",
        "deliverables": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in deliverables
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "output"
    print(json.dumps(build(args.root, output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
