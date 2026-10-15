#!/usr/bin/env python3
"""Download MicroBooNE open-data HDF5 shards from Zenodo with resume + md5 check.

Each Zenodo connection is throttled to ~1 MB/s, so shards are fetched in parallel.
Usage: download_zenodo.py RECORD_ID OUTDIR [--jobs N] [--pattern SUBSTR]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def md5sum(path, bufsize=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def fetch(f, outdir):
    path = os.path.join(outdir, f["key"])
    _, want = f["checksum"].split(":")
    if os.path.exists(path) and os.path.getsize(path) == f["size"] and md5sum(path) == want:
        return f"{f['key']}: ok (cached)"
    for _ in range(5):
        subprocess.run(["curl", "-sSL", "--retry", "10", "-C", "-", "-o", path, f["links"]["self"]])
        if os.path.exists(path) and os.path.getsize(path) == f["size"]:
            break
    got = md5sum(path)
    return f"{f['key']}: {'ok' if got == want else 'MD5 MISMATCH ' + got}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("outdir")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pattern", default=".h5")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{a.record}") as r:
        rec = json.load(r)
    files = sorted((f for f in rec["files"] if a.pattern in f["key"]), key=lambda f: f["key"])
    with open(os.path.join(a.outdir, f"zenodo_{a.record}.json"), "w") as fo:
        json.dump(rec, fo, indent=1)
    print(f"{len(files)} files, {sum(f['size'] for f in files)/1e9:.1f} GB", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(lambda f: fetch(f, a.outdir), files):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
