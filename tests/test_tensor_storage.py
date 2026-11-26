"""Read-only mappings and validated chunked tensor-file access."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import test_arda


def tensor_header(shape, dtype=4, width=8):
    header = bytearray(128)
    header[:8] = b"ARDATEN1"
    header[8:11] = bytes([dtype, len(shape), 1])
    elements = 1
    for i, length in enumerate(shape):
        elements *= length
        struct.pack_into("<Q", header, 32 + i * 8, length)
    struct.pack_into("<QQ", header, 16, 128, elements * width)
    return header


class TensorStorageTests(unittest.TestCase):
    run_code = test_arda.ArdaTests.run_code

    def test_read_only_mapping_and_atomic_replacement(self):
        self.run_code("""
        tol tensor as t
        esta x = t.array([1,2,3,4],"int64",[2,2])
        t.save("data.atn",t.permute(x,[1,0]),160)
        esta mapped = t.mmap("data.atn")
        esta reverse = t.view(mapped,[4],[-1],3)
        mapped = mun
        collect()
        pedo(t.dtype(reverse),t.tolist(reverse))
        pedo(t.tolist(t.read_block("data.atn",1,2)))
        try:
            t.save("data.atn",x,159)
        catch error:
            pedo(startswith(error["message"],"tensor file exceeds disk budget") or len(error["message"])>0)
        t.save("data.atn",t.array([5,6],"int32"))
        pedo(t.tolist(reverse),t.tolist(t.mmap("data.atn")))
        try:
            t.put(reverse,[0],7)
        catch error:
            pedo(len(error["message"])>0)
        pedo(t.tolist(t.read_block("data.atn",2,0)))
        """, "int64 [4, 2, 3, 1]\n[3, 2]\nnae\n[4, 2, 3, 1] [5, 6]\nnae\n[]", stress=True)

    def test_external_header_errors_and_large_bounded_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = tensor_header([1])
            errors = {
                "truncated": valid + b"\x00",
                "oversized": valid + b"\x00" * 9,
                "endian": bytearray(valid) + b"\x00" * 8,
                "offset": bytearray(valid) + b"\x00" * 8,
                "huge_shape": tensor_header([1]) + b"\x00" * 8,
            }
            errors["endian"][10] = 2
            errors["offset"][16] = 64
            struct.pack_into("<Q", errors["huge_shape"], 32, 2**32)
            for name, payload in errors.items():
                (root / name).write_bytes(payload)
            os.mkfifo(root / "fifo")
            n = 16_000_003
            with (root / "large").open("wb") as handle:
                handle.write(tensor_header([n]))
                handle.seek(128 + (n - 2) * 8)
                handle.write(struct.pack("<qq", -125, 9223372036854775807))
            script = root / "read.arda"
            script.write_text("""
tol tensor as t
an path in ["truncated","oversized","endian","offset","huge_shape","fifo"]:
    try:
        t.mmap(path)
    catch error:
        pedo("invalid")
esta m = t.mmap("large")
pedo(t.shape(m),t.at(m,[16000001]))
pedo(t.tolist(t.read_block("large",16000001,2)))
try:
    t.read_block("large",16000002,2)
catch error:
    pedo("bounds")
try:
    t.tolist(m)
catch error:
    pedo("bounded")
t.save("empty",t.zeros([2,0],"uint8"))
pedo(t.shape(t.mmap("empty")),t.bytes(t.mmap("empty")))
""")
            process = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                     capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout.splitlines(), ["invalid"] * 6 + [
                "[16000003] -125", "[-125, 9223372036854775807]", "bounds", "bounded", "[2, 0] 0",
            ])

    def test_disk_budget_and_temporary_cleanup_after_failed_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "destination").mkdir()
            script = root / "save.arda"
            script.write_text("""
tol tensor as t
esta x = t.array([1,2],"int64")
try:
    t.save("destination",x)
catch e:
    pedo("rename error")
try:
    t.save("budget",x,143)
catch e:
    pedo("budget error")
t.save("destination.atn",x,144)
""")
            process = subprocess.run([test_arda.BIN, "--gc-stress", script], cwd=root,
                                     capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout, "rename error\nbudget error\n")
            self.assertEqual({path.name for path in root.iterdir()},
                             {"destination", "destination.atn", "save.arda"})
            self.assertEqual((root / "destination.atn").read_bytes(),
                             tensor_header([2]) + struct.pack("<qq", 1, 2))
