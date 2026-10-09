"""只读本任务私人进程状态/日志元数据；不打印原始会话或凭据。"""
import ctypes
import json
import os
from pathlib import Path

root = Path(os.environ["LOCALAPPDATA"]) / "WaferCodexAutopilot/20261009"
path = root / "dispatch_state.json"
if not path.exists():
    print(json.dumps({"dispatch_exists": False})); raise SystemExit(0)
state = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(state, dict):
    print(json.dumps({"dispatch_exists": True, "dispatch_invalid": True})); raise SystemExit(1)
running = None
if os.name == "nt":
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    dll.OpenProcess.restype = ctypes.c_void_p
    dll.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    dll.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = dll.OpenProcess(0x1000, False, int(state["pid"]))
    if handle:
        code = ctypes.c_uint32()
        if dll.GetExitCodeProcess(handle, ctypes.byref(code)):
            running = code.value == 259
        dll.CloseHandle(handle)
    else:
        running = False
stream = Path(state["stdout_file"])
events = []
if stream.exists():
    # 限量读取尾部；原始输出可能含私有路径，绝不回显。
    with stream.open("rb") as f:
        f.seek(max(0, stream.stat().st_size - 500000))
        for line in f.read().decode("utf-8", errors="replace").splitlines():
            try: event = json.loads(line)
            except (ValueError, TypeError): continue
            if isinstance(event, dict): events.append(event)
tools = []
for event in events:
    msg = event.get("message") or {}
    if isinstance(msg, dict):
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "tool_use":
                tools.append(content.get("name"))
results = [r for r in events if r.get("type") == "result"]
repo = Path(__file__).resolve().parents[3]
receipts = list((repo / "协作/02_ClaudeCode_实操").glob("自动接续交付_20261009_*/接收回执.json"))
out = {"dispatch_exists": True, "session_id": state["session_id"], "pid": state["pid"],
       "process_running": running, "prompt_sha256": state["prompt_sha256"],
       "stream_bytes": stream.stat().st_size if stream.exists() else 0,
       "recent_tool_names": tools[-8:], "has_result": bool(results),
       "result_status": results[-1].get("subtype") if results else None,
       "result_is_error": results[-1].get("is_error") if results else None,
       "receipt_files": [str(p.relative_to(repo)).replace(chr(92), "/") for p in receipts]}
print(json.dumps(out, ensure_ascii=False, indent=2))
