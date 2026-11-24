"""Interleaved wall-clock benchmarks with output verification. Python 3.8+."""
import argparse, pathlib, sys, subprocess, time, statistics, platform, random, os
root = pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline',type=pathlib.Path,help='Path to optimized Arda 0.8.0 executable')
p.add_argument('--rounds',type=int,default=9)
p.add_argument('--cpu',type=int,help='Optional Linux CPU affinity, inherited by children')
p.add_argument('--current',type=pathlib.Path,default=root.parent/'arda')
a=p.parse_args()
if a.rounds<3: p.error('need at least 3 rounds')
if a.cpu is not None: os.sched_setaffinity(0,{a.cpu})
runtimes=[('Arda 0.8.0',str(a.baseline.resolve()),'.arda'),('Arda 0.8.1',str(a.current.resolve()),'.arda'),('CPython',sys.executable,'.py')]
print(platform.platform(), 'Python', platform.python_version())
if hasattr(os,'sched_getaffinity'): print('CPU affinity:',sorted(os.sched_getaffinity(0)))
print(f'Median of {a.rounds} interleaved wall-clock runs incl. startup; milliseconds; stdout verified. One warmup per runtime/workload.')
print('Workload      Arda 0.8.0   Arda 0.8.1      CPython    Speedup     Reduction')
rng=random.Random(42)
for name, expected in [('fib','75025'),('sum','499999500000'),('quest','300000'),('alloc','5000050000')]:
    samples={label:[] for label,_,_ in runtimes}
    def run(runtime):
        label,cmd,ext=runtime
        start=time.perf_counter_ns(); result=subprocess.run([cmd,str(root/(name+ext))],capture_output=True,text=True,check=True,timeout=30); elapsed=(time.perf_counter_ns()-start)/1e6
        if result.stdout.strip()!=expected: raise RuntimeError((label,name,result.stdout))
        return elapsed
    for runtime in runtimes: run(runtime)
    for _ in range(a.rounds):
        order=runtimes[:];rng.shuffle(order)
        for runtime in order: samples[runtime[0]].append(run(runtime))
    old,new,py=[statistics.median(samples[label]) for label,_,_ in runtimes]
    print(f'{name:10} {old:12.3f} {new:12.3f} {py:12.3f} {old/new:9.2f}x {100*(1-new/old):11.1f}%')
    for label,_,_ in runtimes: print(f'  {label:12} raw ms: '+', '.join(f'{x:.3f}' for x in samples[label]))
