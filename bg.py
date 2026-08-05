"""bg.py — minimal background process manager.

Usage:
    python bg.py start <name> <command> [args...]   spawn detached, record PID
    python bg.py stop <name>                        kill process tree (no-op if not running)
    python bg.py status <name>                      exit 0 if running, 1 otherwise
    python bg.py list                               show all managed processes

PID files live in .bg/<name>.pid next to this script; child output goes to
.bg/<name>.log. Callers: .claude/launch_claude.bat and .claude/launch_claude.ps1
(daemons may rewrite their own pid file via --pid-file, same location).
"""
import os
import sys
import time
import subprocess
from pathlib import Path

BG_DIR = Path(__file__).resolve().parent / ".bg"


def pid_file(name: str) -> Path:
    return BG_DIR / f"{name}.pid"


def log_file(name: str) -> Path:
    return BG_DIR / f"{name}.log"


def read_pid(name: str):
    try:
        return int(pid_file(name).read_text().strip())
    except (OSError, ValueError):
        return None


def is_alive(pid) -> bool:
    """True if a process with this pid exists (never signals it)."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(pid: int):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def cmd_start(name: str, command: list) -> int:
    BG_DIR.mkdir(exist_ok=True)
    old = read_pid(name)
    if old and is_alive(old):
        print(f"[bg] {name} already running (pid {old}), restarting", flush=True)
        kill_tree(old)
    flags = 0
    if sys.platform == "win32":
        flags = (subprocess.DETACHED_PROCESS
                 | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW)
    log = open(log_file(name), "ab")
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    except OSError as e:
        print(f"[bg] failed to start {name}: {e}", file=sys.stderr)
        return 1
    finally:
        log.close()
    time.sleep(0.5)  # catch instant crashes (bad path, import error)
    if proc.poll() is not None:
        print(f"[bg] {name} exited immediately (code {proc.returncode}), "
              f"see {log_file(name)}", file=sys.stderr)
        return 1
    pid_file(name).write_text(str(proc.pid))
    print(f"[bg] started {name} (pid {proc.pid})", flush=True)
    return 0


def cmd_stop(name: str) -> int:
    pid = read_pid(name)
    if pid and is_alive(pid):
        kill_tree(pid)
        print(f"[bg] stopped {name} (pid {pid})", flush=True)
    else:
        print(f"[bg] {name} not running", flush=True)
    pid_file(name).unlink(missing_ok=True)
    return 0


def cmd_status(name: str) -> int:
    pid = read_pid(name)
    if pid and is_alive(pid):
        print(f"[bg] {name} running (pid {pid})", flush=True)
        return 0
    print(f"[bg] {name} not running", flush=True)
    return 1


def cmd_list() -> int:
    if not BG_DIR.is_dir():
        print("[bg] no managed processes")
        return 0
    entries = sorted(BG_DIR.glob("*.pid"))
    if not entries:
        print("[bg] no managed processes")
        return 0
    for f in entries:
        name = f.stem
        pid = read_pid(name)
        state = f"running (pid {pid})" if pid and is_alive(pid) else "dead"
        print(f"  {name:20s} {state}")
    return 0


def main(argv) -> int:
    if len(argv) >= 4 and argv[1] == "start":
        return cmd_start(argv[2], argv[3:])
    if len(argv) == 3 and argv[1] == "stop":
        return cmd_stop(argv[2])
    if len(argv) == 3 and argv[1] == "status":
        return cmd_status(argv[2])
    if len(argv) == 2 and argv[1] == "list":
        return cmd_list()
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
