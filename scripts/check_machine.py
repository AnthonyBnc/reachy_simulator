"""Check whether a computer is good enough for reachy_simulator (Windows, macOS or Linux).

Run on the machine you want to check, from the project root, inside the .venv:
    python scripts/check_machine.py
"""

import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "reach_scene.xml"

results = []  # (level, name, detail); level: OK / WARN / FAIL


def report(level, name, detail=""):
    results.append(level)
    print(f"  [{level:<4}] {name}" + (f"  ({detail})" if detail else ""))


def total_ram_gb():
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 1e9
        if platform.system() == "Darwin":
            import subprocess

            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return None


def main():
    print("\n1. Computer")
    print(f"  OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  CPU: {platform.processor() or 'unknown'}")

    cores = os.cpu_count() or 0
    report("OK" if cores >= 8 else "WARN" if cores >= 4 else "FAIL", "CPU cores", f"{cores}; want >= 4, ideally 8+")

    ram = total_ram_gb()
    if ram is None:
        report("WARN", "RAM", "could not read")
    else:
        report("OK" if ram >= 15 else "WARN" if ram >= 7.5 else "FAIL", "RAM", f"{ram:.0f} GB; want >= 8, ideally 16")

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    report("OK" if free_gb >= 20 else "WARN" if free_gb >= 5 else "FAIL", "Free disk", f"{free_gb:.0f} GB; want >= 5, ideally 20")

    print("\n2. Python and packages")
    report("OK" if sys.version_info[:2] == (3, 10) else "FAIL", "Python 3.10 (matches Reachy)", platform.python_version())
    report("OK" if sys.prefix != sys.base_prefix else "WARN", "Running inside a venv", sys.prefix)

    modules = {}
    for name in ["mujoco", "gymnasium", "stable_baselines3", "torch", "numpy", "reachy2_sdk"]:
        try:
            modules[name] = __import__(name)
            report("OK", f"import {name}", getattr(modules[name], "__version__", ""))
        except Exception as e:
            report("FAIL", f"import {name}", f"{type(e).__name__}: {e}")

    if "mujoco" not in modules:
        return
    mujoco = modules["mujoco"]

    print("\n3. MuJoCo")
    if not SCENE.exists():
        report("FAIL", "Scene file", f"{SCENE} not found; run from a full clone of the repo")
        return
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    report("OK", "Load reach_scene.xml", f"nq={model.nq}")

    for _ in range(500):
        mujoco.mj_step(model, data)
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data)
    sps = n / (time.perf_counter() - t0)
    env_sps = sps / 25  # 20 Hz control = 25 physics steps per env step
    report("OK" if env_sps >= 1000 else "WARN" if env_sps >= 300 else "FAIL",
           "Physics speed (1 core)", f"{sps:,.0f} steps/s = {env_sps:,.0f} env steps/s; want >= 1,000")

    try:
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            renderer.update_scene(data)
            image = renderer.render()
        report("OK" if image.mean() > 5 else "WARN", "OpenGL rendering (needed for the viewer and videos)",
               "image is black" if image.mean() <= 5 else "")
    except Exception as e:
        report("WARN", "OpenGL rendering", f"{type(e).__name__}: {e}. Viewer may not work (remote desktop / VM?)")

    if "torch" in modules:
        print("\n4. Training speed")
        torch = modules["torch"]
        net = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(),
                                  torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 7))
        opt = torch.optim.Adam(net.parameters())
        x = torch.randn(256, 64)
        t0 = time.perf_counter()
        for _ in range(500):
            loss = net(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ups = 500 / (time.perf_counter() - t0)
        report("OK" if ups >= 800 else "WARN" if ups >= 250 else "FAIL",
               "CPU network updates (SAC-sized)", f"{ups:,.0f}/s; want >= 800")
        gpu = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else \
            "Apple MPS" if torch.backends.mps.is_available() else "none"
        print(f"  [info] GPU for PyTorch: {gpu} (not needed until camera-based training)")

    print("\n5. Viewer command on this OS")
    if platform.system() == "Darwin":
        print("  macOS: scripts with launch_passive -> mjpython scripts/<script>.py")
    else:
        print("  Windows/Linux: no mjpython needed -> python scripts/<script>.py")


if __name__ == "__main__":
    main()
    fails, warns = results.count("FAIL"), results.count("WARN")
    print()
    if fails:
        print(f"RESULT: NOT READY ({fails} failed, {warns} warnings). Fix the FAIL items above.")
    elif warns:
        print(f"RESULT: USABLE ({warns} warnings). Fine for simulation; check the WARN items.")
    else:
        print("RESULT: GOOD. This computer is ready for reachy_simulator.")
    sys.exit(1 if fails else 0)
