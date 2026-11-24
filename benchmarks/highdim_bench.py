"""Compare identical kernels in two Arda binaries, timing only operation calls."""
import pathlib, subprocess, statistics, sys, platform, os
root=pathlib.Path(__file__).resolve().parents[1]
bins=[str(pathlib.Path(x).resolve()) for x in sys.argv[1:]]
if len(bins)!=2: raise SystemExit('usage: highdim_bench.py OLD_ARDA NEW_ARDA')
if hasattr(os,'sched_setaffinity'):
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});print('CPU affinity:',cpu)
print(platform.platform());print(subprocess.check_output(['cc','--version'],text=True).splitlines()[0])
rows=[[],[]]
for i in range(9):
    for j in ([0,1] if i%2==0 else [1,0]):
        raw=subprocess.check_output([bins[j],str(root/'benchmarks/highdim_perf.arda')],text=True)
        d,m,c,*ts=map(float,raw.split()); assert (d,m,c)==(3000000,512,2),(d,m,c)
        rows[j].append([1000*x for x in ts])
meds=[[statistics.median(x[k] for x in r) for k in range(3)] for r in rows]
for j in range(2):
    print(subprocess.check_output([bins[j],'--version'],text=True).strip(), 'samples ms:',rows[j])
for k,name in enumerate(['dot 2M','matmul 512x512','strided copy 2M']):
    a,b=meds[0][k],meds[1][k];print(f'{name}: {a:.3f} -> {b:.3f} ms, {a/b:.3f}x')
