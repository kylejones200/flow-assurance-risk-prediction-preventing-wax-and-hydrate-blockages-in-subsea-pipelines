#!/usr/bin/env python3
import time, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/"src"))
from compute_kernel import generate_telemetry, logistic_risk_probability
def main():
    t0=time.perf_counter()
    for _ in range(200):
        t=generate_telemetry(4000,77)
        logistic_risk_probability(*t)
    py_s=time.perf_counter()-t0
    try:
        import flow_assurance_risk_prediction_preventing_wax_and_hydrate_blockages_in_subsea_pipelines_rs as rs
    except ImportError:
        print("Build rust extension"); return
    rs_s=rs.bench_kernel_py(4000,77,200)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s/max(rs_s,1e-9):.1f}x")
    pt=generate_telemetry(100,77)
    py_prob=logistic_risk_probability(*pt)
    rs_prob=rs.logistic_risk_probability_py(*[np.ascontiguousarray(x,dtype=float) for x in pt])
    np.testing.assert_allclose(py_prob, np.asarray(rs_prob), rtol=1e-10)
    print("Correctness: OK")
if __name__=="__main__": main()
