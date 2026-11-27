#!/usr/bin/env bash
# ESMFold RNGD web service on 127.0.0.1 (default port 8090).
#   service/run_service.sh start [app.py options]   e.g. start --port 8091
#   service/run_service.sh stop | status | log
# From your own PC:  ssh -N -L 8090:127.0.0.1:8090 <user>@<server>   then open http://localhost:8090
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PID_FILE=logs/service.pid
LOG_FILE=logs/service.log
PORT=${PORT:-8090}

running() { [[ -f $PID_FILE ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

case "${1:-start}" in
  start)
    if running; then echo "already running (pid $(cat "$PID_FILE"))"; exit 0; fi
    mkdir -p logs
    nohup .venv/bin/python -u service/app.py --port "$PORT" "${@:2}" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "started pid $(cat "$PID_FILE") -> http://127.0.0.1:$PORT  (log: $LOG_FILE)"
    ;;
  stop)
    if ! running; then echo "not running"; rm -f "$PID_FILE"; exit 0; fi
    kill "$(cat "$PID_FILE")"
    for _ in $(seq 1 30); do running || break; sleep 1; done
    running && kill -9 "$(cat "$PID_FILE")"
    rm -f "$PID_FILE"
    echo "stopped"
    ;;
  status)
    if running; then
      echo "running (pid $(cat "$PID_FILE"))"
      curl -s -m 5 "http://127.0.0.1:$PORT/api/status" | .venv/bin/python -c \
        'import json,sys; s=json.load(sys.stdin); print({k: s[k] for k in ("state","mode","devices","served","waiting","ready_seconds")})'
    else
      echo "not running"
    fi
    ;;
  log)
    tail -n 50 "$LOG_FILE"
    ;;
  *)
    echo "usage: $0 {start|stop|status|log} [app.py options]"; exit 2
    ;;
esac
