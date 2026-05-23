use flow_assurance_risk_prediction_preventing_wax_and_hydrate_blockages_in_subsea_pipelines_core::{generate_telemetry, logistic_risk_probability};
fn main() { for _ in 0..200 { let t=generate_telemetry(4000,77); let _=logistic_risk_probability(&t.wat_celsius,&t.temp_in_celsius,&t.temp_out_celsius,&t.water_cut,&t.inhibitor_active,&t.shear_proxy,&t.heavy_waxy); } }
