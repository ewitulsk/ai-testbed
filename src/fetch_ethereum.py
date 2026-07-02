"""
Concurrent multi-endpoint fetch of pre-merge Ethereum block/uncle data.

Uncles (ommers) are stale blocks: valid blocks that lost the canonical race but
were later referenced. They are the densest public record of *organic* forks on
a major PoW chain. Per canonical block we record (number, timestamp, n_uncles);
per uncle we record its inclusion distance = nephew_height - uncle_height (the
fork-depth analog, protocol-capped at 6 in Ethereum).

Public RPC endpoints throttle a single connection to ~45 blk/s, so we fan out
across several endpoints with a thread pool. Output:
    data/eth_blocks.csv.gz   number,timestamp,n_uncles
    data/eth_uncles.csv.gz   nephew,uncle_number,distance
"""
import argparse, gzip, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.mevblocker.io",
    "https://1rpc.io/eth",
    "https://eth-mainnet.public.blastapi.io",
    "https://eth.rpc.blxrbdn.com",
]
CA = "/root/.ccr/ca-bundle.crt"
os.environ.setdefault("REQUESTS_CA_BUNDLE", CA)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

_thread_local = None


def _session():
    import threading
    global _thread_local
    if _thread_local is None:
        _thread_local = threading.local()
    s = getattr(_thread_local, "s", None)
    if s is None:
        s = requests.Session()
        _thread_local.s = s
    return s


def rpc_batch(payload, ep_index, timeout=20, retries=6):
    s = _session()
    last = None
    for attempt in range(retries):
        url = ENDPOINTS[(ep_index + attempt) % len(ENDPOINTS)]
        try:
            r = s.post(url, json=payload, timeout=timeout)
            if r.status_code != 200:
                last = f"HTTP {r.status_code} @ {url}"
                time.sleep(min(1.5 ** attempt, 8))
                continue
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                last = data["error"]
                time.sleep(min(1.5 ** attempt, 8))
                continue
            return data
        except Exception as e:  # noqa
            last = f"{type(e).__name__} @ {url}"
            time.sleep(min(1.5 ** attempt, 8))
    raise RuntimeError(f"batch failed: {last}")


def fetch_header_chunk(args):
    lo, hi, ep_index = args
    payload = [{"jsonrpc": "2.0", "method": "eth_getBlockByNumber",
                "params": [hex(b), False], "id": b} for b in range(lo, hi)]
    data = rpc_batch(payload, ep_index)
    by_id = {r["id"]: r for r in data if "id" in r}
    rows = []            # (number, ts, n_uncles)
    uncle_refs = []      # (nephew, idx)
    missing = [b for b in range(lo, hi) if b not in by_id or not by_id[b].get("result")]
    # retry any missing individually
    for b in missing:
        single = rpc_batch({"jsonrpc": "2.0", "method": "eth_getBlockByNumber",
                            "params": [hex(b), False], "id": b}, ep_index + 1)
        by_id[b] = single
    for b in range(lo, hi):
        res = by_id[b]["result"]
        ts = int(res["timestamp"], 16)
        unc = res.get("uncles", []) or []
        rows.append((b, ts, len(unc)))
        for idx in range(len(unc)):
            uncle_refs.append((b, idx))
    return rows, uncle_refs


def fetch_uncle_chunk(args):
    refs, ep_index = args  # refs: list of (nephew, idx)
    payload = [{"jsonrpc": "2.0",
                "method": "eth_getUncleByBlockNumberAndIndex",
                "params": [hex(neph), hex(idx)], "id": i}
               for i, (neph, idx) in enumerate(refs)]
    data = rpc_batch(payload, ep_index)
    by_id = {r["id"]: r for r in data if "id" in r}
    out = []
    for i, (neph, idx) in enumerate(refs):
        r = by_id.get(i)
        if not r or not r.get("result"):
            continue
        unum = int(r["result"]["number"], 16)
        out.append((neph, unum, neph - unum))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=12_400_000)
    ap.add_argument("--end", type=int, default=13_400_000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    chunks = []
    i = 0
    for lo in range(args.start, args.end, args.batch):
        hi = min(lo + args.batch, args.end)
        chunks.append((lo, hi, i % len(ENDPOINTS)))
        i += 1

    print(f"[phase1] {args.end - args.start} blocks in {len(chunks)} chunks, "
          f"{args.workers} workers over {len(ENDPOINTS)} endpoints", flush=True)
    all_rows = []
    all_uncle_refs = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_header_chunk, c): c for c in chunks}
        for fut in as_completed(futs):
            try:
                rows, refs = fut.result()
            except Exception as e:
                print(f"  chunk {futs[fut][:2]} failed: {e}", flush=True)
                continue
            all_rows.extend(rows)
            all_uncle_refs.extend(refs)
            done += 1
            if done % 200 == 0 or done == len(chunks):
                rate = len(all_rows) / (time.time() - t0)
                print(f"  {done}/{len(chunks)} chunks, {len(all_rows)} blocks, "
                      f"{len(all_uncle_refs)} uncles, {rate:.0f} blk/s", flush=True)

    all_rows.sort()
    bpath = os.path.join(DATA, "eth_blocks.csv.gz")
    with gzip.open(bpath, "wt") as f:
        f.write("number,timestamp,n_uncles\n")
        for (num, ts, nu) in all_rows:
            f.write(f"{num},{ts},{nu}\n")
    print(f"[phase1] wrote {len(all_rows)} blocks -> {bpath}", flush=True)

    # phase 2: uncle distances
    print(f"[phase2] fetching {len(all_uncle_refs)} uncle headers", flush=True)
    uchunks = []
    UB = 150
    for j in range(0, len(all_uncle_refs), UB):
        uchunks.append((all_uncle_refs[j:j + UB], (j // UB) % len(ENDPOINTS)))
    uncle_rows = []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_uncle_chunk, c): c for c in uchunks}
        d2 = 0
        for fut in as_completed(futs):
            try:
                uncle_rows.extend(fut.result())
            except Exception as e:
                print(f"  uncle chunk failed: {e}", flush=True)
            d2 += 1
            if d2 % 50 == 0 or d2 == len(uchunks):
                print(f"  {d2}/{len(uchunks)} uncle chunks, "
                      f"{len(uncle_rows)} done", flush=True)
    uncle_rows.sort()
    upath = os.path.join(DATA, "eth_uncles.csv.gz")
    with gzip.open(upath, "wt") as f:
        f.write("nephew,uncle_number,distance\n")
        for (neph, unum, dist) in uncle_rows:
            f.write(f"{neph},{unum},{dist}\n")
    print(f"[done] {len(all_rows)} blocks, {len(uncle_rows)} uncles in "
          f"{time.time() - t0:.0f}s (phase2 {time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
