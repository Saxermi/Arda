"""Compare packed Q4/Q8 products and prediction quality on a local CPU."""
import argparse
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import tempfile

from cpu_baseline import THREAD_ENV, save_tensor


def program(rows, inner, columns, rounds, iterations, encoding, method):
    if encoding:
        setup = f'''
let source = q.quantize(weights,"{encoding}")
weights = mun
collect()
let payload = q.info(source)["bytes"]
'''
    else:
        setup = '''
let source = weights
let payload = t.bytes(source)
'''
    if method == "reconstructed":
        setup += '''
source = q.dequantize(source)
collect()
'''
    if method in ("dense", "reconstructed"):
        product = "t.matmul(source,dense_input)"
        if columns == 1:
            product = f"t.reshape({product},[{rows}])"
    else:
        operation = "gemv" if columns == 1 else "gemm"
        product = f'q.{operation}(source,input,"{method}")'
    serialized_weights = ('t.save("rebuilt.atn",q.dequantize(source))'
                          if encoding and method != "reconstructed"
                          else 't.save("rebuilt.atn",source)')
    activation = f't.reshape(dense_input,[{inner}])' if columns == 1 else 'dense_input'
    return f"""
import tensor as t
import quant as q
import compute as c
let weights = t.mmap("weights.atn")
let dense_input = t.mmap("input.atn")
let input = {activation}
{setup}
let result = {product}
for trial in range({rounds}):
    let elapsed = 0.0
    for i in range({iterations}):
        collect()
        let start = monotonic()
        result = {product}
        elapsed = elapsed + (monotonic()-start)
    print(elapsed/{iterations})
print(payload,c.memory_stats()["peak_budgeted_bytes"])
t.save("output.atn",result)
{serialized_weights}
"""


def run(np, binary, root, weights, activation, rounds, iterations, encoding, method):
    rows, inner = weights.shape
    columns = activation.shape[1]
    source = root / "quantization.arda"
    source.write_text(program(rows, inner, columns, rounds, iterations, encoding, method),
                      encoding="utf-8")
    result = subprocess.check_output([binary, source], cwd=root, text=True).splitlines()
    samples = [float(line) for line in result[:-1]]
    payload, peak = map(int, result[-1].split())
    output = np.fromfile(root / "output.atn", dtype="<f8", offset=128).reshape(rows, columns)
    rebuilt = np.fromfile(root / "rebuilt.atn", dtype="<f8", offset=128).reshape(rows, inner)
    np.testing.assert_allclose(output, rebuilt @ activation, rtol=4e-12, atol=4e-12)
    return samples, payload, peak, output, rebuilt


def relative_error(np, result, reference):
    return np.linalg.norm(result - reference) / np.linalg.norm(reference)


def performance(np, binary, root, weights, activation, rounds, iterations):
    save_tensor(root / "weights.atn", weights)
    save_tensor(root / "input.atn", activation)
    expected = weights @ activation
    shape = f"{weights.shape} × {activation.shape}"
    print(f"Benchmark {shape}, median over {rounds} batches after one warm-up")
    for label, encoding, method in [
        ("dense", None, "dense"),
        ("q8-reference", "q8", "reference"),
        ("q8-auto", "q8", "auto"),
        ("q8-rebuilt-dense", "q8", "reconstructed"),
        ("q4-reference", "q4", "reference"),
        ("q4-auto", "q4", "auto"),
        ("q4-rebuilt-dense", "q4", "reconstructed"),
    ]:
        samples, payload, peak, output, rebuilt = run(
            np, binary, root, weights, activation, rounds, iterations, encoding, method)
        middle = median(samples)
        mad = median(abs(sample - middle) for sample in samples)
        if method == "reconstructed":
            payload = rebuilt.nbytes
        print(f"{label:16s} {1e3*middle:8.4f} ms  MAD {1e3*mad:.4f} ms  "
              f"weights {payload:9d} B  heap peak {peak:9d} B  "
              f"output relative L2 {relative_error(np, output, expected):.4%}", flush=True)


def classifier(np, binary, root, rng):
    weights = rng.normal(size=(16, 128))
    inputs = rng.normal(size=(128, 1024))
    logits = weights @ inputs
    labels = logits.argmax(axis=0)
    save_tensor(root / "weights.atn", weights)
    save_tensor(root / "input.atn", inputs)
    print("Synthetic classifier with 16 classes, 128 inputs and 1,024 held-out examples")
    for encoding in ("q8", "q4"):
        _, _, _, prediction, rebuilt = run(np, binary, root, weights, inputs, 1, 1, encoding, "auto")
        agreement = int(np.count_nonzero(prediction.argmax(axis=0) == labels))
        print(f"{encoding}  weight relative L2 {relative_error(np, rebuilt, weights):.4%}  "
              f"logit relative L2 {relative_error(np, prediction, logits):.4%}  "
              f"top-1 agreement {agreement}/{labels.size}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "arda")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--inner", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.inner, args.columns, args.rounds, args.iterations) < 1:
        parser.error("dimensions and repetitions must be positive")
    if max(args.rows * args.inner, args.inner * args.columns, args.rows * args.columns) > 16_000_000:
        parser.error("tensor shape exceeds the materialization limit")
    os.sched_setaffinity(0, {args.cpu})
    os.environ.update(THREAD_ENV)
    import numpy as np

    binary = args.binary.resolve()
    print(platform.platform(), "CPU", args.cpu, "NumPy", np.__version__,
          subprocess.check_output([binary, "--version"], text=True).strip(), flush=True)
    rng = np.random.default_rng(180300)
    weights = rng.normal(size=(args.rows, args.inner))
    with tempfile.TemporaryDirectory(prefix="arda-quant-") as directory:
        root = Path(directory)
        for columns in (1, args.columns) if args.columns != 1 else (1,):
            activation = rng.normal(size=(args.inner, columns))
            performance(np, binary, root, weights, activation, args.rounds, args.iterations)
        classifier(np, binary, root, rng)


if __name__ == "__main__":
    main()
