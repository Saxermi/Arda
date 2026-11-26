"""Benchmark Arda's 1,000-step checkpoint replay on Linux."""
import argparse
from array import array
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tempfile

STEPS = 1000
CHECKPOINT_EVERY = 50
INITIAL = (0.008, 0.012, 0.017, 0.021, 0.025, 0.031, 0.038, 0.045)
TOLERANCE = 2e-12


def workload(rows, checkpoint):
    body = f"""
    if (i+1) % {CHECKPOINT_EVERY} == 0:
        state = ad.checkpoint(state,boundary)
        boundary = state
        collect()
""" if checkpoint else ""
    return f"""
import ad
import tensor as t
import compute as c

let x = ad.variable(t.zeros([{rows},{len(INITIAL)}],"float64") + t.array({list(INITIAL)},"float64"))
let state = x
let boundary = x
collect()
let start = monotonic()
for i in range({STEPS}):
    state = t.sin(state)
{body.rstrip()}
let loss = t.sum(state)
state = mun
boundary = mun
collect()
let forward = monotonic() - start
let before = monotonic()
ad.backward(loss,mun,{{"retain":[x],"consume":true}})
let backward = monotonic() - before
let stats = c.memory_stats()
let replay = ad.offload_stats(ad.offload_plan(".",0))["vm_checkpoint_replay_operations"]
print(ad.value(loss),forward,backward,stats["peak_budgeted_bytes"],replay)
t.save("gradient.atn",ad.grad(x))
"""


def reference(rows):
    values, gradients = [], []
    for initial in INITIAL:
        state, slope = initial, 1.0
        for _ in range(STEPS):
            slope *= math.cos(state)
            state = math.sin(state)
        values.append(state)
        gradients.append(slope)
    return math.fsum(values) * rows, gradients


def read_gradient(path, count):
    with path.open("rb") as source:
        header = source.read(128)
        if not header.startswith(b"ARDATEN1"):
            raise ValueError(f"unexpected Arda tensor header in {path}")
        payload = source.read()
    if len(payload) != count * 8:
        raise ValueError(f"expected {count} Float64 gradients in {path}")
    gradient = array("d")
    gradient.frombytes(payload)
    if sys.byteorder != "little":
        gradient.byteswap()
    return gradient


def run(binary, program, directory):
    with tempfile.TemporaryFile(mode="w+") as out:
        with subprocess.Popen([binary, program], cwd=directory, stdout=out) as process:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, [binary, program])
        out.seek(0)
        value, forward, backward, peak, replay = out.read().split()
    return float(value), float(forward), float(backward), int(peak), usage.ru_maxrss, int(replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "arda")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    args = parser.parse_args()
    count = args.rows * len(INITIAL)
    if not 1 <= count <= 16_000_000 or args.rounds < 1 or args.warmups < 0:
        parser.error("rows must fit the tensor limit, rounds must be positive, and warmups nonnegative")
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    binary = args.binary.resolve()
    loss_reference, gradient_reference = reference(args.rows)
    samples = {"ordinary": [], "checkpoint": []}
    max_mode_error = 0.0
    print(platform.platform(), subprocess.check_output([binary, "--version"], text=True).strip())
    print(f"CPU {args.cpu}, {STEPS} steps, {count} Float64 entries, {CHECKPOINT_EVERY}-step checkpoints")
    with tempfile.TemporaryDirectory(prefix="arda-vilya-a1-") as tmp:
        directory = Path(tmp)
        programs = {}
        for mode in samples:
            programs[mode] = directory / f"{mode}.arda"
            programs[mode].write_text(workload(args.rows, mode == "checkpoint"), encoding="utf-8")

        for index in range(args.rounds + args.warmups):
            pair = {}
            order = tuple(samples) if index % 2 == 0 else tuple(reversed(samples))
            for mode in order:
                value, forward, backward, peak, rss, replay = run(binary, programs[mode], directory)
                gradient = read_gradient(directory / "gradient.atn", count)
                width = len(gradient_reference)
                error = max(
                    abs(component - gradient_reference[i % width]) if math.isfinite(component) else math.inf
                    for i, component in enumerate(gradient)
                )
                if not math.isclose(value, loss_reference, rel_tol=TOLERANCE) or error > TOLERANCE:
                    raise ValueError(f"{mode} failed the loss or gradient check: {value}, {error}")
                if replay != (count * STEPS if mode == "checkpoint" else 0):
                    raise ValueError(f"{mode} produced unexpected replay count {replay}")
                pair[mode] = gradient
                if index >= args.warmups:
                    samples[mode].append((forward, backward, peak, rss, replay, error))
                print(f"round {index} {mode}: forward {forward:.3f}s, backward {backward:.3f}s, "
                      f"peak tensor {peak} B, peak RSS {rss} KiB, replay {replay}, "
                      f"gradient reference error {error:.3g}", flush=True)
            error = max(abs(a - b) for a, b in zip(pair["ordinary"], pair["checkpoint"]))
            max_mode_error = max(max_mode_error, error)
            if error > TOLERANCE:
                raise ValueError(f"checkpoint gradient differs from ordinary by {error}")

    for mode, runs in samples.items():
        fw, bw, peak, rss, replay, error = zip(*runs)
        print(f"{mode}: median forward {median(fw):.3f}s [{min(fw):.3f}, {max(fw):.3f}], "
              f"backward {median(bw):.3f}s [{min(bw):.3f}, {max(bw):.3f}], "
              f"peak tensor {median(peak) / 2**20:.2f} MiB, peak RSS {median(rss) / 1024:.2f} MiB, "
              f"replay {median(replay):.0f}, max gradient reference error {max(error):.3g}")
    print(f"Max absolute gradient difference between modes: {max_mode_error:.3g}")


if __name__ == "__main__":
    main()
