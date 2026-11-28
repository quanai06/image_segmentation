"""Canh nhiệt máy Mac: đọc trạng thái nhiệt của macOS và tạm dừng khi máy quá nóng.

Trạng thái (NSProcessInfo.thermalState): 0 bình thường, 1 hơi ấm, 2 nóng (bắt đầu hạ xung), 3 rất nóng.
Không cần sudo. Trên máy không phải macOS trả -1 và không bao giờ chặn.
"""

import platform
import subprocess
import time

from medseg.utils.logging import get_logger

log = get_logger()
LEVELS = {-1: "không rõ", 0: "bình thường", 1: "hơi ấm", 2: "nóng", 3: "rất nóng"}
_SCRIPT = "ObjC.import('Foundation'); $.NSProcessInfo.processInfo.thermalState"


def thermal_state() -> int:
    if platform.system() != "Darwin":
        return -1
    try:
        out = subprocess.run(["osascript", "-l", "JavaScript", "-e", _SCRIPT], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return -1


def wait_until_cool(max_level: int = 1, poll_sec: int = 60, cooldown_sec: int = 0) -> int:
    """Kiểm tra nhiệt; nếu vượt max_level thì ngủ tới khi nguội. Trả về trạng thái lúc đo đầu tiên."""
    first = state = thermal_state()
    if state > max_level:
        log.warning(f"Máy {LEVELS[state]} (mức {state}) -> tạm dừng, kiểm tra lại mỗi {poll_sec}s")
        while state > max_level:
            time.sleep(poll_sec)
            state = thermal_state()
        log.info(f"Máy đã nguội (mức {state}), chạy tiếp")
    if cooldown_sec:
        time.sleep(cooldown_sec)
    return first
