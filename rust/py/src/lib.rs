use flow_assurance_risk_prediction_preventing_wax_and_hydrate_blockages_in_subsea_pipelines_core::{generate_telemetry, logistic_risk_probability};
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (n_segments, seed=77))]
fn generate_telemetry_py<'py>(py: Python<'py>, n_segments: usize, seed: u64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let t = generate_telemetry(n_segments, seed);
    Ok((t.wat_celsius.into_pyarray(py), t.temp_in_celsius.into_pyarray(py), t.temp_out_celsius.into_pyarray(py),
        t.water_cut.into_pyarray(py), t.inhibitor_active.into_pyarray(py), t.shear_proxy.into_pyarray(py), t.heavy_waxy.into_pyarray(py)))
}

#[pyfunction]
fn logistic_risk_probability_py<'py>(py: Python<'py>, wat: PyReadonlyArray1<f64>, tin: PyReadonlyArray1<f64>, tout: PyReadonlyArray1<f64>, wc: PyReadonlyArray1<f64>, inhib: PyReadonlyArray1<f64>, shear: PyReadonlyArray1<f64>, hw: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(logistic_risk_probability(wat.as_slice()?, tin.as_slice()?, tout.as_slice()?, wc.as_slice()?, inhib.as_slice()?, shear.as_slice()?, hw.as_slice()?).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (n_segments=4000, seed=77, iterations=200))]
fn bench_kernel_py(n_segments: usize, seed: u64, iterations: usize) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let t = generate_telemetry(n_segments, seed);
        let _ = logistic_risk_probability(&t.wat_celsius, &t.temp_in_celsius, &t.temp_out_celsius, &t.water_cut, &t.inhibitor_active, &t.shear_proxy, &t.heavy_waxy);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn flow_assurance_risk_prediction_preventing_wax_and_hydrate_blockages_in_subsea_pipelines_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_telemetry_py, m)?)?;
    m.add_function(wrap_pyfunction!(logistic_risk_probability_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
