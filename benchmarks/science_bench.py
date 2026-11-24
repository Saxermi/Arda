"""Median of seven in-program timings; construction is excluded."""
import os, pathlib, statistics, subprocess, sys, platform
root = pathlib.Path(__file__).resolve().parents[1]
print(platform.platform(), platform.machine())
print('GCC:',subprocess.check_output(['cc','--version'],text=True).splitlines()[0])
for binary in sys.argv[1:] or ['./arda']:
    rows=[]
    for i in range(7):
        a,b=subprocess.check_output([str(pathlib.Path(binary).resolve()),str(root/'benchmarks/science.arda')],text=True).splitlines()
        s,t,scalar,native=map(float,a.split()); result,mat,backend=b.split()
        assert s==t==2000000 and float(result)==256
        rows.append([scalar,native,float(mat)])
    meds=[statistics.median(row[i] for row in rows)*1000 for i in range(3)]
    print(f'{backend}: scalar={meds[0]:.3f} ms, dot={meds[1]:.3f} ms, matmul={meds[2]:.3f} ms, scalar/dot={meds[0]/meds[1]:.1f}x')
    print('samples (seconds):',rows)
