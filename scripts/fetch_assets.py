"""Download author-hosted experiment assets; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
ASSETS = {
    "weights": ("https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
                "7bee48f3476c75616c3c5721ab256ff8", "weights"),
    "data": ("https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip", None, "data"),
}


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def fetch(name):
    url, md5, destination = ASSETS[name]
    archive = ROOT / "downloads" / url.rsplit("/", 1)[1]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    if md5 and digest(archive, "md5") != md5:
        raise ValueError(f"Official MD5 mismatch: {archive}. Remove this archive and retry.")
    destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        # Validate every path before extracting author-hosted archives.
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        zipped.extractall(destination)
    record = {"url": url, "sha256": digest(archive), "md5": digest(archive, "md5")}
    (destination / "download.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", choices=list(ASSETS), default=list(ASSETS))
    for asset in parser.parse_args().assets:
        fetch(asset)
