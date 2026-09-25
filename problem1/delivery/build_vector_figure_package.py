"""Bundle every figure embedded in the Problem 1 paper as SVG and PDF."""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path


DELIVERY = Path(__file__).resolve().parent
PROJECT = DELIVERY.parent
PAPER = DELIVERY / "problem1_paper_integrated.md"
BUNDLE = DELIVERY / "problem1_vector_figures"
FIGURES = BUNDLE / "figures"
ZIP = DELIVERY / "problem1_vector_figures.zip"

SOURCES = {
    "fig_feature_label": PROJECT / "outputs" / "figures",
    "fig_modality_quality_academic": PROJECT / "outputs" / "figures",
    "fig_alignment_similarity": PROJECT / "outputs" / "figures",
    "typical_alignment": DELIVERY,
    "problem1_feature_label_distribution": PROJECT / "outputs",
    "sample_000_timeline": PROJECT / "outputs",
    "smoke_timeline": PROJECT / "outputs",
}


def main() -> None:
    markdown = PAPER.read_text(encoding="utf-8")
    embedded = re.findall(r"!\[[^]]*\]\(([^)]+)\)", markdown)
    assert len(embedded) == len(SOURCES), (len(embedded), len(SOURCES))
    assert all(Path(link).suffix == ".svg" for link in embedded)
    assert {Path(link).stem for link in embedded} == set(SOURCES)
    assert (BUNDLE / "图表放置说明.md").is_file()

    FIGURES.mkdir(parents=True, exist_ok=True)
    digests = []
    for stem, source_dir in SOURCES.items():
        for suffix in ("svg", "pdf"):
            source = source_dir / f"{stem}.{suffix}"
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            destination = FIGURES / source.name
            shutil.copy2(source, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            digests.append(f"{digest}  figures/{source.name}")

    (BUNDLE / "SHA256SUMS.txt").write_text("\n".join(digests) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUNDLE))
    with zipfile.ZipFile(ZIP) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"corrupt ZIP entry: {bad}")
        assert len(archive.namelist()) == 16, archive.namelist()
    print(f"created {ZIP} with 14 vector figures, placement guide, and checksums")


if __name__ == "__main__":
    main()
