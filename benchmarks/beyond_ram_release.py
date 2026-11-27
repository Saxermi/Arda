#!/usr/bin/env python3
"""Run the Linux large-file reduction checks on a fully written tensor fixture."""

import argparse
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import tempfile

MIB = 1024 * 1024
BLOCK_ELEMENTS = 65536
ADDRESS_LIMIT = 128 * MIB

STATS = """\
import compute as c
c.memory_budget(4*1024*1024)
let result = c.stream_stats("fixture.atn",65536)
print(result["count"],result["mean"],result["variance"])
"""

REDUCE = """\
import compute as c
import tensor as t
c.memory_budget(4*1024*1024)
def add(total,block):
    return total+t.sum(block)
print(c.reduce_file("fixture.atn",0,add,65536))
"""


def write_fixture(path, mib):
    count = mib * MIB // 8
    header = bytearray(128)
    header[:11] = b"ARDATEN1" + bytes([8, 2, 1])
    struct.pack_into("<QQQQ", header, 16, 128, 8 * count,
                     count // BLOCK_ELEMENTS, BLOCK_ELEMENTS)
    payload = struct.pack("<4d", 1, 2, 3, 4) * (MIB // 32)
    with path.open("wb") as file:
        file.write(header)
        for _ in range(mib):
            file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    info = path.stat()
    if info.st_blocks * 512 < info.st_size:
        raise RuntimeError("fixture requires uncompressed, allocated file data")
    print(f"fixture: {info.st_size} bytes, {info.st_blocks * 512} allocated, "
          f"address-space limit: {ADDRESS_LIMIT} bytes", flush=True)
    return count


def run_pass(binary, time_bin, directory, program, expected):
    output = directory / "time.txt"
    command = [time_bin, "-f", "%e %M %I", "-o", str(output), "prlimit",
               f"--as={ADDRESS_LIMIT}:{ADDRESS_LIMIT}", "--", str(binary), program]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{program} failed ({result.returncode}): {result.stderr.strip()}")
    values = result.stdout.split()
    if len(values) != len(expected) or any(
        not math.isclose(float(value), numeric, rel_tol=0.0, abs_tol=1e-10)
        for value, numeric in zip(values, expected)
    ):
        raise RuntimeError(f"{program}: expected {expected}, received {values}")
    seconds, rss, inputs = map(float, output.read_text().split())
    return seconds, int(rss), int(inputs)


def measure(binary, time_bin, directory, name, source, expected, repeats):
    program = name + ".arda"
    (directory / program).write_text(source, encoding="utf-8")
    with (directory / "fixture.atn").open("rb") as fixture:
        os.posix_fadvise(fixture.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    first = run_pass(binary, time_bin, directory, program, expected)
    warm = [run_pass(binary, time_bin, directory, program, expected) for _ in range(repeats)]
    timings, peaks, reads = zip(*warm)
    print(f"{name}: verified {expected}")
    print(f"  eviction requested: {first[0]:.2f} s, {first[1]} KiB RSS, {first[2]} FS inputs")
    print(f"  warm: {statistics.median(timings):.2f} s median "
          f"[{min(timings):.2f}, {max(timings):.2f}], {max(peaks)} KiB max RSS, "
          f"{min(reads)}–{max(reads)} FS inputs ({repeats} runs)")
    # GNU time reports RSS in KiB. The reduction and statistics gates differ.
    rss_limit = (512 if name == "stats" else 1024) * 1024
    if max(first[1], *peaks) >= rss_limit:
        raise RuntimeError(f"{name} exceeded its RSS gate of {rss_limit} KiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arda", type=Path, default=Path("./arda"))
    parser.add_argument("--directory", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--mib", type=int, default=2048)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--operation", choices=("both", "stats", "reduce"), default="both")
    parser.add_argument("--time", default="/usr/bin/time", help="path to GNU time")
    args = parser.parse_args()
    if args.mib < 1 or args.warm_runs < 1:
        parser.error("--mib and --warm-runs must be positive")
    if len(Path("/proc/swaps").read_text().splitlines()) != 1:
        parser.error("the no-swap gate requires a Linux environment without active swap")
    if shutil.disk_usage(args.directory).free < args.mib * MIB + 128:
        parser.error("the fixture needs disk space for its full payload and header")
    binary = args.arda.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="arda-beyond-ram-", dir=args.directory) as tmp:
        root = Path(tmp)
        count = write_fixture(root / "fixture.atn", args.mib)
        if args.operation in ("stats", "both"):
            measure(binary, args.time, root, "stats", STATS, [count, 2.5, 1.25], args.warm_runs)
        if args.operation in ("reduce", "both"):
            measure(binary, args.time, root, "reduce", REDUCE, [count * 5 // 2], args.warm_runs)


if __name__ == "__main__":
    main()
