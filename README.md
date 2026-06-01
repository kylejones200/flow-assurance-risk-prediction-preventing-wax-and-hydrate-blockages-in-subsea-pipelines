# Flow Assurance Risk Prediction Preventing Wax and Hydrate Blockages in Subsea Pipelines

**Published:** 2025-10-07
**Medium:** [https://medium.com/@kyle-t-jones/flow-assurance-risk-prediction-preventing-wax-and-hydrate-blockages-in-subsea-pipelines-3b78edadb561](https://medium.com/@kyle-t-jones/flow-assurance-risk-prediction-preventing-wax-and-hydrate-blockages-in-subsea-pipelines-3b78edadb561)

## Business context

BP's Thunder Horse platform experienced wax deposition that reduced pipeline flow capacity by 40% which lead to lost production cost exceeded $100 million and a months long remediation. Oil and gas operators are implement machine learning-based flow assurance monitoring to gain early warning capabilities that prevent blockages, optimize chemical injection, and avoid multi-million dollar interventions.

Flow assurance risk prediction is about understanding the complex interaction between thermal margins, flow regime, fluid composition, and inhibitor effectiveness to identify which pipeline segments will develop wax or hydrate problems before they impact production. Machine learning techniques can process real-time telemetry to predict risk with sufficient lead time for preventive action.

Subsea oil and gas production operates in one of the most challenging flow assurance environments. Cold seawater temperatures (4--15°C), high pressures (50--200 bar), and long tiebacks (5--50 km) create ideal conditions for wax precipitation and hydrate formation. A single blockage can shut in production for weeks, cost $50--200 million in lost revenue, and require expensive intervention vessels.



## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — telemetry features and logistic wax/hydrate risk. Reference PyO3 benchmark: **~1.5×** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p flow_assurance_risk_prediction_preventing_wax_and_hydrate_blockages_in_subsea_pipelines_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).