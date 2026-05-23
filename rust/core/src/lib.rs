//! Pipeline telemetry generation and logistic risk from thermal margin.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as u32
    }

    fn uniform(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryArrays {
    pub wat_celsius: Vec<f64>,
    pub temp_in_celsius: Vec<f64>,
    pub temp_out_celsius: Vec<f64>,
    pub water_cut: Vec<f64>,
    pub inhibitor_active: Vec<f64>,
    pub shear_proxy: Vec<f64>,
    pub heavy_waxy: Vec<f64>,
}

pub fn generate_telemetry(n_segments: usize, seed: u64) -> TelemetryArrays {
    let mut rng = Lcg::new(seed);
    let mut wat_celsius = Vec::with_capacity(n_segments);
    let mut temp_in_celsius = Vec::with_capacity(n_segments);
    let mut temp_out_celsius = Vec::with_capacity(n_segments);
    let mut water_cut = Vec::with_capacity(n_segments);
    let mut inhibitor_active = Vec::with_capacity(n_segments);
    let mut shear_proxy = Vec::with_capacity(n_segments);
    let mut heavy_waxy = Vec::with_capacity(n_segments);

    for _ in 0..n_segments {
        let hw = if rng.uniform() < 0.15 { 1.0 } else { 0.0 };
        let wat = if hw > 0.0 {
            35.0 + rng.normal() * 2.0
        } else {
            27.0 + rng.normal() * 2.0
        };
        let tin = 32.0 + rng.normal() * 4.0;
        let drop = (3.0 + rng.normal() * 2.0).clamp(0.5, 10.0);
        let tout = tin - drop;
        let pressure = 55.0 + rng.normal() * 6.0;
        let flow = (2.0 + rng.normal() * 0.6).clamp(0.3, 4.0);
        let shear = 0.3 * flow / (pressure / 50.0);
        let wc = {
            // Beta(2,10) approximate from uniforms
            let a = rng.uniform().powf(1.0 / 2.0);
            let b = rng.uniform().powf(1.0 / 10.0);
            a / (a + b).max(1e-9)
        };
        let inhib = if rng.uniform() < 0.3 { 1.0 } else { 0.0 };

        wat_celsius.push(wat);
        temp_in_celsius.push(tin);
        temp_out_celsius.push(tout);
        water_cut.push(wc);
        inhibitor_active.push(inhib);
        shear_proxy.push(shear);
        heavy_waxy.push(hw);
    }

    TelemetryArrays {
        wat_celsius,
        temp_in_celsius,
        temp_out_celsius,
        water_cut,
        inhibitor_active,
        shear_proxy,
        heavy_waxy,
    }
}

pub fn logistic_risk_probability(
    wat_celsius: &[f64],
    temp_in_celsius: &[f64],
    temp_out_celsius: &[f64],
    water_cut: &[f64],
    inhibitor_active: &[f64],
    shear_proxy: &[f64],
    heavy_waxy: &[f64],
) -> Vec<f64> {
    let n = wat_celsius.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let avg_temp = 0.5 * (temp_in_celsius[i] + temp_out_celsius[i]);
        let thermal_margin = wat_celsius[i] - avg_temp;
        let mut logit = 0.0;
        if thermal_margin > 0.0 {
            logit += 0.8;
        }
        if thermal_margin > 2.0 {
            logit += 0.4;
        }
        if water_cut[i] > 0.2 {
            logit += 0.3;
        }
        if heavy_waxy[i] > 0.5 {
            logit += 0.3;
        }
        logit -= 0.4 * inhibitor_active[i];
        if shear_proxy[i] > 0.9 {
            logit -= 0.3;
        }
        logit -= 0.4;
        let p = 1.0 / (1.0 + (-logit).exp());
        out.push(p);
    }
    out
}
