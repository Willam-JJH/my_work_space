"""
Proxy Daemon (Supervisor) — production-grade supervisor
=======================================================
- waitress WSGI server (multi-threaded, production)
- Auto-restart on crash (exponential backoff, max 30s)
- Graceful shutdown on SIGINT/SIGTERM
- PID file management (for bg.py integration)
"""
import sys, os, time, signal, argparse, socket
from pathlib import Path

_pid_file = None

def _write_pid():
    if _pid_file:
        _pid_file.parent.mkdir(parents=True, exist_ok=True)
        _pid_file.write_text(str(os.getpid()))

def _cleanup_pid():
    if _pid_file and _pid_file.exists():
        _pid_file.unlink(missing_ok=True)

_shutdown = False

def _on_signal(signum, frame):
    global _shutdown
    name = signal.Signals(signum).name
    print(f"\n[Daemon] Received {name}, shutting down...", flush=True)
    _shutdown = True
    _cleanup_pid()
    sys.exit(0)

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)
import atexit
atexit.register(_cleanup_pid)

def run_daemon(host="127.0.0.1", port=4000, threads=8,
               max_restarts=10, restart_window=60):
    from waitress import serve
    from proxy import app

    _write_pid()

    print("=" * 55)
    print("  Proxy Daemon (waitress)")
    print(f"  Listening: http://{host}:{port}")
    print(f"  Workers: {threads} threads")
    print(f"  Auto-restart: ON (max {max_restarts} per {restart_window}s)")
    print("=" * 55)

    restart_times = []
    restart_count = 0

    while not _shutdown:
        try:
            print(f"[Daemon] Starting waitress (attempt #{restart_count+1})...", flush=True)
            serve(app, host=host, port=port, threads=threads,
                  connection_limit=100, channel_timeout=300)
        except Exception as e:
            if _shutdown:
                break
            print(f"[Daemon] waitress crashed: {type(e).__name__}: {e}", flush=True)

        if _shutdown:
            break

        now = time.time()
        restart_times = [t for t in restart_times if now - t < restart_window]
        restart_times.append(now)
        if len(restart_times) > max_restarts:
            print(f"[Daemon] {len(restart_times)} crashes in {restart_window}s, giving up", flush=True)
            _cleanup_pid()
            return 1

        restart_count += 1
        delay = min(2 ** min(restart_count, 5), 30)
        print(f"[Daemon] Auto-restart in {delay}s...", flush=True)
        time.sleep(delay)

    print("[Daemon] Exited", flush=True)
    _cleanup_pid()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proxy Daemon")
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--restart-window", type=int, default=60)
    args = parser.parse_args()

    _pid_file = args.pid_file

    proxy_path = Path(__file__).resolve().parent
    if str(proxy_path) not in sys.path:
        sys.path.insert(0, str(proxy_path))

    sys.exit(run_daemon(
        host=args.host, port=args.port, threads=args.threads,
        max_restarts=args.max_restarts, restart_window=args.restart_window,
    ))
