"""ESMFold-on-RNGD web service: sequence in, 3D structure (PDB + coordinates) out.

Stdlib HTTP server. Binds 127.0.0.1 by default; reach it from another machine
with an SSH tunnel:  ssh -N -L 8090:127.0.0.1:8090 <user>@<server>

  GET  /                    web UI (3Dmol.js viewer)
  GET  /api/status          service state, NPU status (furiosa-smi)
  POST /api/fold            {"sequence": "..."} -> {"id": ...}
  GET  /api/jobs/<id>       state / progress / result (PDB, coordinates, pLDDT)
  GET  /api/jobs/<id>/pdb   predicted structure as a PDB file
"""
from __future__ import annotations

import argparse
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
WARMUP_SEQUENCE = "MQIFVKTLTGKTITLEV"
MAX_JOBS = 200
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png",
                 ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8"}


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class Service:
    def __init__(self, args):
        self.args = args
        self.state, self.error = "loading", None
        self.engine = None
        self.jobs: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.pending: queue.Queue[dict] = queue.Queue()
        self.lock = threading.Lock()
        self.started = time.time()
        self.ready_seconds = None
        self.served = 0
        self._npu = (0.0, None)
        threading.Thread(target=self._worker, name="esmfold-worker", daemon=True).start()

    # All model work happens on this one thread: it owns the RNGD compile-cache context
    # and serialises NPU access.
    def _worker(self):
        args = self.args
        try:
            start = time.monotonic()
            log("loading facebook/esmfold_v1 ...")
            self.engine = E.Engine(use_npu=not args.cpu, devices=[f"furiosa:{int(i)}" for i in args.devices.split(",")],
                                   recycles=args.recycles, threads=args.threads, cache_dir=args.cache)
            self.engine.enter_context()
            self.state = "warming"
            self.engine.prepare_npu()
            log("warm-up fold" + ("" if args.cpu else " (loads the compiled RNGD binary from the cache)") + " ...")
            self.engine.fold(WARMUP_SEQUENCE)
            self.ready_seconds = time.monotonic() - start
            self.state = "ready"
            log(f"ready in {self.ready_seconds:.1f}s (mode={'cpu' if args.cpu else 'rngd-hybrid'})")
        except Exception as exc:  # noqa: BLE001 - surfaced through /api/status
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            log(traceback.format_exc())
            return
        while True:
            job = self.pending.get()
            self.engine.progress = self.engine.new_progress()  # Don't show the previous job's progress.
            job["state"], job["started"] = "running", time.time()
            try:
                job["result"] = self.engine.fold(job["sequence"])
                job["state"] = "done"
                self.served += 1
                r = job["result"]
                log(f"job {job['id']} {r['length']} aa: pLDDT {r['mean_plddt']:.1f}, {r['timings']['total']:.2f}s")
            except Exception as exc:  # noqa: BLE001 - reported to the client
                job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
                log(traceback.format_exc())
            job["finished"] = time.time()

    def submit(self, text):
        if self.state == "error":
            raise RuntimeError(f"서비스 오류: {self.error}")
        seq = E.clean_sequence(text)
        job = {"id": uuid.uuid4().hex[:12], "sequence": seq, "state": "queued", "created": time.time(),
               "started": None, "finished": None, "result": None, "error": None}
        with self.lock:
            self.jobs[job["id"]] = job
            while len(self.jobs) > MAX_JOBS:
                self.jobs.popitem(last=False)
        self.pending.put(job)
        log(f"job {job['id']} queued ({len(seq)} aa)")
        return job

    def job_view(self, job):
        view = {k: job[k] for k in ("id", "sequence", "state", "error", "created", "started", "finished")}
        if job["state"] == "running":
            view["progress"] = dict(self.engine.progress)
            view["elapsed"] = time.time() - job["started"]
        elif job["state"] == "queued":
            with self.lock:
                view["queue_ahead"] = sum(1 for j in self.jobs.values()
                                          if j["state"] in ("queued", "running") and j["created"] < job["created"])
            view["service_state"] = self.state
        elif job["state"] == "done":
            view["result"] = job["result"]
        return view

    def npu_status(self):
        stamp, cached = self._npu
        if cached is not None and time.monotonic() - stamp < 2:
            return cached
        try:
            run = lambda *a: json.loads(subprocess.run(["furiosa-smi", *a, "--format=json"], capture_output=True,
                                                       text=True, timeout=5, check=True).stdout)
            info = {d["dev_name"]: d for d in run("info")}
            procs = run("ps")
            resident = self.engine.npu_resident if self.engine else {}
            devices = []
            for d in run("status"):
                pes = d["pe_utilizations"]
                dram = d["memory"]["DRAM"]
                meta = info.get(d["device"], {})
                owners = [p["pid"] for p in procs if re.search(rf"\b{d['device']}:", p.get("dev_name", ""))]
                ours = resident.get(f"furiosa:{d['device'].removeprefix('npu')}")  # fusion=8: one chip per index
                devices.append({
                    "name": d["device"], "liveness": d["liveness"],
                    "mem_used_gib": dram["used_size"] / 2**30, "mem_total_gib": dram["total_size"] / 2**30,
                    "resident_gib": ours["bytes"] / 2**30 if ours else None,
                    "layers": f"{min(ours['layers'])}-{max(ours['layers'])}" if ours else None,
                    "pe_util": sum(p["pe_utilization"] for p in pes) / max(len(pes), 1),
                    "pe_occupied": sum(bool(p["pe_occupancy"]) for p in pes), "pe_total": len(pes),
                    "occupied": bool(owners), "owned_by_service": os.getpid() in owners,
                    "temperature": meta.get("temperature"), "power": meta.get("power"),
                    "firmware": meta.get("firmware"),
                })
            result = {"devices": devices}
        except Exception as exc:  # noqa: BLE001
            result = {"devices": [], "error": f"{type(exc).__name__}: {exc}"}
        self._npu = (time.monotonic(), result)
        return result

    def status(self):
        engine = self.engine
        with self.lock:
            waiting = sum(1 for j in self.jobs.values() if j["state"] in ("queued", "running"))
        return {
            "state": self.state, "error": self.error,
            "model": "facebook/esmfold_v1",
            "parameters_b": round(engine.parameters_b, 3) if engine else None,
            "mode": "cpu" if self.args.cpu else "rngd-hybrid",
            "devices": [f"furiosa:{int(i)}" for i in self.args.devices.split(",")] if not self.args.cpu else [],
            "min_len": E.MIN_LEN, "max_len": E.MAX_LEN, "bucket_tokens": E.BUCKET_TOKENS,
            "recycles": self.args.recycles, "ready_seconds": self.ready_seconds,
            "uptime_seconds": time.time() - self.started, "served": self.served, "waiting": waiting,
            "progress": dict(engine.progress) if engine else None,
            "npu": self.npu_status(),
        }


SERVICE: Service


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return  # Client went away mid-response (closed tab, aborted poll).
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "ESMFold-RNGD/1.0"

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/api/status", "/api/jobs", "/static/")):
            log(f"{self.address_string()} {fmt % args}")

    def _send(self, code, body, content_type, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _file(self, path):
        path = path.resolve()
        if STATIC not in path.parents or not path.is_file():
            return self._json(404, {"error": "not found"})
        self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if path.startswith("/static/"):
            return self._file(STATIC / path[len("/static/"):])
        if path == "/api/status":
            return self._json(200, SERVICE.status())
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(/pdb)?", path)
        if match:
            job = SERVICE.jobs.get(match[1])
            if job is None:
                return self._json(404, {"error": "unknown job"})
            if not match[2]:
                return self._json(200, SERVICE.job_view(job))
            if job["state"] != "done":
                return self._json(409, {"error": f"job is {job['state']}"})
            name = f"esmfold_rngd_{job['sequence'][:12]}_{job['id']}.pdb"
            return self._send(200, job["result"]["pdb"].encode(), "chemical/x-pdb",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/fold":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > 100_000:
            return self._json(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": '요청 본문이 JSON이 아닙니다. 예: {"sequence": "GYDPETGTWG"}'})
        if not isinstance(body, dict):
            return self._json(400, {"error": '요청 본문은 {"sequence": "..."} 형태여야 합니다.'})
        try:
            job = SERVICE.submit(str(body.get("sequence", "")))
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except RuntimeError as exc:
            return self._json(503, {"error": str(exc)})
        self._json(202, {"id": job["id"], "sequence": job["sequence"]})


def main():
    global SERVICE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--devices", default="0,1", help="RNGD chips (each fused to 8 PEs)")
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cache", type=Path, default=E.ROOT / "cache/service")
    parser.add_argument("--cpu", action="store_true", help="CPU only (no RNGD), for debugging")
    args = parser.parse_args()
    SERVICE = Service(args)
    server = Server((args.host, args.port), Handler)
    log(f"listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
