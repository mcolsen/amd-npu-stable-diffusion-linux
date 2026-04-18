"""Runtime health probes and per-step telemetry for NPU inference.

All observers here are read-only sysfs/ioctl queries (no sudo). The telemetry
CSV is written to $XDNA_CACHE_DIR/telemetry/<run_id>.csv (default
~/.cache/xdna2-npu-diffusion/telemetry), flushed after each row so it survives
an abrupt process or system crash.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    from nputop.ioctl import NpuDevice
except ImportError:
    NpuDevice = None  # telemetry will degrade to sysfs-only

# Below this, the NPU is almost certainly PM-suspended or power-capped. 1200 MHz
# is ~66% of Strix Halo's 1800 MHz max; normal inference sits at or near max.
MIN_EXPECTED_NPU_MHZ = 1200

# iGPU busy % above which we warn about probable contention.
GPU_BUSY_WARN_PCT = 20.0


def _default_telemetry_root() -> Path:
    override = os.environ.get("XDNA_CACHE_DIR")
    if override:
        return Path(override) / "telemetry"
    return Path.home() / ".cache" / "xdna2-npu-diffusion" / "telemetry"


_TELEMETRY_ROOT = _default_telemetry_root()

_FIELDS = [
    "t_unix", "phase", "step", "step_wall_s", "attempt",
    "gpu_busy_pct", "gpu_ppt_w", "gpu_temp_c", "gpu_freq_mhz", "vddgfx_mv",
    "cpu_temp_c",
    "npu_mp_mhz", "npu_h_mhz", "npu_tops_curr", "npu_task_curr",
    "npu_pm_status", "npu_power_mode",
    "npu_ctx_errors", "npu_ctx_submissions", "npu_ctx_completions",
]


# ---------------------------------------------------------------------------
# sysfs discovery
# ---------------------------------------------------------------------------

def _find_drm_card_with_busy() -> Optional[Path]:
    for path in sorted(Path("/sys/class/drm").glob("card*")):
        # Skip connector entries (card0-HDMI-A-1 etc.)
        if "-" in path.name:
            continue
        busy = path / "device" / "gpu_busy_percent"
        if busy.exists():
            return path / "device"
    return None


def _find_hwmon_by_name(name: str) -> Optional[Path]:
    for path in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (path / "name").read_text().strip() == name:
                return path
        except OSError:
            continue
    return None


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Preflight: GPU contention, render-node users, NPU wake + clock probe
# ---------------------------------------------------------------------------

def sample_gpu_busy(samples: int = 5, interval_s: float = 0.1) -> Optional[float]:
    """Average iGPU busy-percent over `samples` reads. None if sysfs unreadable."""
    card = _find_drm_card_with_busy()
    if card is None:
        return None
    readings: list[int] = []
    for i in range(samples):
        if i:
            time.sleep(interval_s)
        v = _read_int(card / "gpu_busy_percent")
        if v is not None:
            readings.append(v)
    return sum(readings) / len(readings) if readings else None


def list_render_node_users() -> list[tuple[int, str]]:
    """Enumerate (pid, comm) of processes holding any /dev/dri/renderD* fd.

    Skips our own pid. Returns deduplicated list."""
    seen: dict[int, str] = {}
    my_pid = os.getpid()
    for fd_dir in glob.glob("/proc/*/fd"):
        parts = fd_dir.split("/")
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        pid = int(parts[2])
        if pid == my_pid or pid in seen:
            continue
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("/dev/dri/renderD"):
                try:
                    comm = Path(f"/proc/{pid}/comm").read_text().strip()
                except OSError:
                    comm = "?"
                seen[pid] = comm
                break
    return sorted(seen.items())


def wake_npu() -> Optional["NpuDevice"]:
    """Open /dev/accel/accel0 and return a held handle.

    Opening the device wakes it from PM suspend; holding the fd keeps autosuspend
    from dropping it mid-run. Returns None if nputop or the device is unavailable.
    """
    if NpuDevice is None:
        return None
    try:
        return NpuDevice()
    except OSError as e:
        print(f"Note: could not open NPU for telemetry ({e})", file=sys.stderr)
        return None


def probe_npu_clock(dev: "NpuDevice", attempts: int = 3, settle_s: float = 0.2) -> int:
    """Return current MP-NPU clock, retrying briefly if it looks PM-suspended."""
    mhz = 0
    for i in range(attempts):
        if i:
            time.sleep(settle_s)
        mhz = dev.query_clocks().mp_npu_mhz
        if mhz >= MIN_EXPECTED_NPU_MHZ:
            break
    return mhz


def preflight(npu: Optional["NpuDevice"], interactive: Optional[bool] = None) -> None:
    """Print GPU contention + NPU clock status before inference starts.

    If `interactive` is None, auto-detect: prompt to continue only when stdin is
    a TTY. Non-interactive runs always continue after printing the warning.
    """
    if interactive is None:
        interactive = sys.stdin.isatty()

    if npu is not None:
        mhz = probe_npu_clock(npu)
        res = npu.query_resource_info()
        pm = npu.query_runtime_pm().status
        mode = npu.query_power_mode()
        print(f"NPU: {mhz}/{res.clk_max_mhz} MHz, mode {mode}, PM {pm}")
        if mhz < MIN_EXPECTED_NPU_MHZ:
            print(
                f"  WARNING: NPU clock {mhz} MHz is below expected "
                f"({MIN_EXPECTED_NPU_MHZ}+). First kernel may hit watchdog."
            )
    else:
        print("NPU: telemetry unavailable (nputop not installed)")

    busy = sample_gpu_busy()
    if busy is None:
        return
    print(f"iGPU busy: {busy:.0f}%")
    if busy < GPU_BUSY_WARN_PCT:
        return
    users = list_render_node_users()
    if users:
        print("  Render-node users: " + ", ".join(f"{c}({p})" for p, c in users))
    print(
        f"  WARNING: iGPU at {busy:.0f}% — NPU may contend for memory bandwidth "
        f"and power budget, risking firmware watchdog timeouts."
    )
    if interactive:
        try:
            reply = input("  Continue anyway? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply != "y":
            print("Aborted by user.")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Per-step telemetry CSV
# ---------------------------------------------------------------------------

class TelemetryWriter:
    """Append-flush CSV writer for per-step telemetry.

    One file per run. `record(...)` snapshots sysfs + NPU ioctl state and writes
    one row; the file is flushed after every write so partial data survives a
    crash. Safe to use when `npu` is None — NPU columns will be blank.
    """

    def __init__(
        self,
        npu: Optional["NpuDevice"],
        run_id: Optional[str] = None,
        root: Path = _TELEMETRY_ROOT,
    ):
        self.run_id = run_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{self.run_id}.csv"
        self._fh = self.path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=_FIELDS)
        self._w.writeheader()
        self._fh.flush()

        self._drm_card = _find_drm_card_with_busy()
        self._hwmon_amdgpu = _find_hwmon_by_name("amdgpu")
        self._hwmon_k10 = _find_hwmon_by_name("k10temp")
        self._npu = npu
        self._my_pid = os.getpid()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _snapshot(self) -> dict:
        row: dict = {k: "" for k in _FIELDS}

        if self._drm_card is not None:
            v = _read_int(self._drm_card / "gpu_busy_percent")
            if v is not None:
                row["gpu_busy_pct"] = v

        if self._hwmon_amdgpu is not None:
            h = self._hwmon_amdgpu
            ppt = _read_int(h / "power1_average")
            if ppt is not None:
                row["gpu_ppt_w"] = round(ppt / 1_000_000, 2)
            t = _read_int(h / "temp1_input")
            if t is not None:
                row["gpu_temp_c"] = round(t / 1000, 1)
            f = _read_int(h / "freq1_input")
            if f is not None:
                row["gpu_freq_mhz"] = f // 1_000_000
            v = _read_int(h / "in0_input")
            if v is not None:
                row["vddgfx_mv"] = v

        if self._hwmon_k10 is not None:
            t = _read_int(self._hwmon_k10 / "temp1_input")
            if t is not None:
                row["cpu_temp_c"] = round(t / 1000, 1)

        if self._npu is not None:
            try:
                c = self._npu.query_clocks()
                row["npu_mp_mhz"] = c.mp_npu_mhz
                row["npu_h_mhz"] = c.h_clock_mhz
            except OSError:
                pass
            try:
                r = self._npu.query_resource_info()
                row["npu_tops_curr"] = r.tops_curr
                row["npu_task_curr"] = r.task_curr
            except OSError:
                pass
            try:
                row["npu_pm_status"] = self._npu.query_runtime_pm().status
            except OSError:
                pass
            try:
                row["npu_power_mode"] = self._npu.query_power_mode()
            except OSError:
                pass
            try:
                mine = [c for c in self._npu.query_hw_contexts() if c.pid == self._my_pid]
                if mine:
                    row["npu_ctx_errors"] = sum(c.errors for c in mine)
                    row["npu_ctx_submissions"] = sum(c.command_submissions for c in mine)
                    row["npu_ctx_completions"] = sum(c.command_completions for c in mine)
            except OSError:
                pass

        return row

    def record(self, phase: str, step: int, step_wall_s: float, attempt: int = 1) -> None:
        row = self._snapshot()
        row["t_unix"] = round(time.time(), 3)
        row["phase"] = phase
        row["step"] = step
        row["step_wall_s"] = round(step_wall_s, 4)
        row["attempt"] = attempt
        self._w.writerow(row)
        self._fh.flush()


# ---------------------------------------------------------------------------
# Retry wrapper for ORT sessions
# ---------------------------------------------------------------------------

def is_npu_timeout_error(exc: BaseException) -> bool:
    """Heuristic match for NPU firmware watchdog / DD runtime timeout errors."""
    msg = str(exc).upper()
    return (
        "ERT_CMD_STATE_TIMEOUT" in msg
        or "UNORDERED_MAP::AT" in msg
        or "TIMED OUT" in msg
        or "TIMEOUT" in msg
    )


def run_with_retry(
    session,
    outputs,
    inputs,
    label: str,
    npu: Optional["NpuDevice"] = None,
    retry_sleep_s: float = 0.5,
):
    """Call session.run() with one retry on NPU-timeout-family errors.

    On failure, logs the NPU clock and hwctx error count (if available) so we
    have a post-mortem anchor, then retries once. Deterministic UNet/VAE inputs
    make retry safe — scheduler state hasn't advanced yet.
    """
    for attempt in (1, 2):
        try:
            return session.run(outputs, inputs)
        except Exception as e:
            if attempt == 2 or not is_npu_timeout_error(e):
                raise
            print(f"\n  {label}: NPU timeout — {str(e)[:140]}")
            if npu is not None:
                try:
                    c = npu.query_clocks()
                    print(f"  Post-failure NPU clock: {c.mp_npu_mhz} MHz")
                except OSError:
                    pass
                try:
                    mine = [x for x in npu.query_hw_contexts() if x.pid == os.getpid()]
                    if mine:
                        print(f"  HW-ctx errors so far: {sum(x.errors for x in mine)}")
                except OSError:
                    pass
            print(f"  Retrying once after {retry_sleep_s}s...")
            time.sleep(retry_sleep_s)
