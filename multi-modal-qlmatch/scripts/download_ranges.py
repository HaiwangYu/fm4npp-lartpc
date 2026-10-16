#!/usr/bin/env python3
"""Finish partially downloaded Zenodo files using parallel HTTP byte ranges (Zenodo throttles per connection).

For each file: the bytes already present are kept; the remainder is split into K ranges fetched in
parallel into <file>.r<i>, then appended in order and the md5 verified.
Usage: download_ranges.py RECORD_ID OUTDIR [--k 4] [--jobs 64]
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from download_zenodo import md5sum


def fetch_range(url, part, a, b):
    """Download bytes [a, b] (inclusive) into `part`, resuming if partially present."""
    want = b - a + 1
    for _ in range(20):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            break
        with open(part, "ab") as fo:
            subprocess.run(["curl", "-sSL", "--retry", "10", "-r", f"{a + have}-{b}", url], stdout=fo)
    return os.path.getsize(part) == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("outdir")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--tag", default="r", help="range-file suffix letter")
    a = ap.parse_args()
    rec = json.load(open(os.path.join(a.outdir, f"zenodo_{a.record}.json")))
    files = sorted((f for f in rec["files"] if f["key"].endswith(".h5")), key=lambda f: f["key"])
    jobs = []
    for f in files:
        path = os.path.join(a.outdir, f["key"])
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= f["size"]:
            continue
        rem = f["size"] - have
        step = -(-rem // a.k)
        for i in range(a.k):
            lo = have + i * step
            hi = min(f["size"], lo + step) - 1
            if lo <= hi:
                jobs.append((f, path, f"{path}.{a.tag}{i}", lo, hi))
    # finish files one at a time: order by remaining bytes, assemble as soon as a file's ranges are done
    def remaining(path):
        return sum(j[4] - j[3] + 1 - (os.path.getsize(j[2]) if os.path.exists(j[2]) else 0)
                   for j in jobs if j[1] == path)
    order = sorted({j[1] for j in jobs}, key=remaining)
    jobs.sort(key=lambda j: (order.index(j[1]), j[3]))
    print(f"{len(jobs)} range jobs", flush=True)
    ok = True
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {path: [ex.submit(fetch_range, j[0]["links"]["self"], j[2], j[3], j[4]) for j in jobs if j[1] == path]
                for path in order}
        for path in order:
            if all(fu.result() for fu in futs[path]):
                assemble([f for f in files if os.path.join(a.outdir, f["key"]) == path][0], a.outdir, jobs)
            else:
                ok = False
    return 0 if ok else 1


def assemble(f, outdir, jobs):
    path = os.path.join(outdir, f["key"])
    parts = sorted((j for j in jobs if j[1] == path), key=lambda j: j[3])
    if parts:
        with open(path, "ab") as fo:
            for j in parts:
                assert os.path.getsize(path) == j[3], (path, os.path.getsize(path), j[3])
                with open(j[2], "rb") as fi:
                    while chunk := fi.read(1 << 26):
                        fo.write(chunk)
                fo.flush()
        for j in parts:
            os.remove(j[2])
    _, want = f["checksum"].split(":")
    got = md5sum(path)
    print(f"{f['key']}: {'ok' if got == want else 'MD5 MISMATCH'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
