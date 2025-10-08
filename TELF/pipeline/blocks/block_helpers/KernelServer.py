import os, signal, subprocess, pathlib, shlex, time

class KernelTiedServer:
    def __init__(self, cmd, *, cwd=None, env=None, log_dir=None):
        """
        cmd: list or string (string is shell-split)
        cwd: working dir
        env: dict of env vars
        log_dir: if set, stdout/stderr are appended to files here
        """
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.log_dir = pathlib.Path(log_dir).expanduser() if log_dir else None
        self.proc = None
        self.watchdog = None
        self.stdout_f = None
        self.stderr_f = None

    def start(self):
        if self.proc and self.running:
            raise RuntimeError("Already running")
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.stdout_f = open(self.log_dir / "server.out", "ab", buffering=0)
            self.stderr_f = open(self.log_dir / "server.err", "ab", buffering=0)

        # 1) start the server in its own process group
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdout=self.stdout_f or subprocess.DEVNULL,
            stderr=self.stderr_f or subprocess.DEVNULL,
            preexec_fn=os.setpgrp,  # new PGID == proc.pid (POSIX)
        )

        # 2) watchdog: when kernel PID disappears, kill the whole PGID
        kernel_pid = os.getpid()
        pgid = self.proc.pid
        script = f"""
        while kill -0 {kernel_pid} 2>/dev/null; do sleep 2; done
        kill -TERM -{pgid} 2>/dev/null
        sleep 5
        kill -KILL -{pgid} 2>/dev/null
        """
        self.watchdog = subprocess.Popen(
            ["bash", "-c", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return self

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        if not self.proc:
            return "not started"
        rc = self.proc.poll()
        return f"running (pid {self.proc.pid}, pgid {self.proc.pid})" if rc is None else f"exited rc={rc}"

    def stop(self, sig=signal.SIGTERM, hard_after=5):
        if not self.proc:
            return
        if self.running:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass
            # optional hard kill after grace
            t0 = time.time()
            while self.running and time.time() - t0 < hard_after:
                time.sleep(0.2)
            if self.running:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # stop watchdog
        if self.watchdog and self.watchdog.poll() is None:
            try:
                self.watchdog.terminate()
            except Exception:
                pass
        # close logs
        for f in (self.stdout_f, self.stderr_f):
            try:
                f and f.close()
            except Exception:
                pass

    def tail(self, n=50, which="out"):
        if not self.log_dir:
            print("(no logs: set log_dir=...)")
            return
        path = self.log_dir / ( "server.out" if which=="out" else "server.err" )
        try:
            with open(path, "rb") as f:
                print(b"".join(f.readlines()[-n:]).decode(errors="replace"))
        except FileNotFoundError:
            print("(no log yet)")

# --- Example ---
# srv = KernelTiedServer(["zsh", "../../Lynx/start_lynx.sh"], log_dir="~/.logs/lynx").start()
# print(srv.status()); srv.tail(100)         # view last 100 lines of stdout
# srv.stop()                                 # stop manually
