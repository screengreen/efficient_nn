
import torch

torch.backends.cudnn.benchmark = False        # main results: PyTorch's default heuristics choose the kernel
torch.backends.cudnn.allow_tf32 = False       # so that FP32 means FP32 on Ampere and newer GPUs
torch.backends.cuda.matmul.allow_tf32 = False



import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, 2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, 1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(256, 256, 3, 2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(256, 512, 1, padding=0, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.max_pool(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


model = SmallCNN()
model = model.cuda().eval()



try:
    import pynvml
except ImportError:
    !pip install -q nvidia-ml-py
    import pynvml

import threading
import time


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

name = pynvml.nvmlDeviceGetName(handle)
print(name.decode() if isinstance(name, bytes) else name)

try:
    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    has_energy_counter = True
except pynvml.NVMLError:
    has_energy_counter = False

print("energy counter" if has_energy_counter else "power sampling fallback")


class EnergyMeter:
    def __init__(self, handle, period=0.005):
        self.handle = handle
        self.period = period

    def _collect(self):
        previous = time.perf_counter()

        while not self.done.is_set():
            time.sleep(self.period)
            now = time.perf_counter()
            watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1e3
            self.joules += watts * (now - previous)
            previous = now

    def start(self):
        if has_energy_counter:
            self.start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return

        self.joules = 0.0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self):
        if has_energy_counter:
            end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return (end_mj - self.start_mj) / 1e3

        self.done.set()
        self.thread.join()
        return self.joules


meter = EnergyMeter(handle)

meter.start()
time.sleep(2.0)
idle_power = meter.stop() / 2.0

print(f"idle power: {idle_power:.1f} W")



import csv
import gc
import random
import time

import numpy as np
import torch


base_sizes = [32, 64, 128, 224, 256, 384, 512, 768, 1024]
base_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

random.seed(42)

extra_sizes = set(random.sample(
    [s for s in range(32, 513, 16) if s not in base_sizes],
    4,
))
extra_batches = set(random.sample(
    [b for b in range(1, 257) if b not in base_batches and b & (b - 1) != 0],
    3,
))

sizes = sorted(set(base_sizes) | extra_sizes)
batches = sorted(set(base_batches) | extra_batches)
points = [(S, B) for S in sizes for B in batches]


def likely_oom(S, B):
    """Skip shapes whose first conv cannot fit, before the allocator fragments."""
    free, _ = torch.cuda.mem_get_info()
    s_out = (S + 2 * 3 - 7) // 2 + 1
    activations = 4 * B * (3 * S * S + 32 * s_out * s_out)
    return 3 * activations >= free


def sustain(S, B, seconds):
    x = torch.randn(B, 3, S, S, device="cuda")
    with torch.inference_mode():
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            model(x)
        torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()


for S, B in ((512, 128), (512, 64), (384, 64)):
    if not likely_oom(S, B):
        print(f"warmup S={S} B={B} for 10s", flush=True)
        sustain(S, B, 10)
        break

results = []

for i, (S, B) in enumerate(points, 1):
    is_validation = S in extra_sizes or B in extra_batches
    print(f"{i}/{len(points)}  S={S} B={B}  val={is_validation}", flush=True)

    x = None

    try:
        if likely_oom(S, B):
            print("    skip, estimated OOM")
            results.append([S, B, None, "OOM", None, is_validation])
            continue

        x = torch.randn(B, 3, S, S, device="cuda")

        with torch.inference_mode():
            for _ in range(5):
                model(x)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        times = []

        with torch.inference_mode():
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        latency = float(np.median(times))
        memory = int(torch.cuda.max_memory_allocated())

        iterations = int(np.clip(0.5 / latency, 5, 2000))

        torch.cuda.synchronize()
        meter.start()

        with torch.inference_mode():
            for _ in range(iterations):
                model(x)

        torch.cuda.synchronize()
        energy = meter.stop() / iterations

        results.append([S, B, latency, memory, energy, is_validation])
        print(
            f"    {latency * 1e3:.2f} ms, {memory / 2**20:.0f} MiB, "
            f"{energy * 1e3:.2f} mJ ({energy / latency:.0f} W)"
        )

    except torch.cuda.OutOfMemoryError:
        print("    OOM")
        results.append([S, B, None, "OOM", None, is_validation])
        gc.collect()

    finally:
        if x is not None:
            del x
        torch.cuda.empty_cache()

with open("measurements.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["S", "B", "latency", "memory", "energy", "is_validation"])
    writer.writerows(results)

measured = sum(row[2] is not None for row in results)
print(f"wrote measurements.csv: {measured} measured, {len(results) - measured} skipped")