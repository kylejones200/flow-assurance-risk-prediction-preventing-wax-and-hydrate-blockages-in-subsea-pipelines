import numpy as np

def generate_telemetry(n_segments=4000, seed=77):
    rng = np.random.default_rng(seed)
    hw = (rng.random(n_segments) < 0.15).astype(float)
    wat = np.where(hw > 0, rng.normal(35, 2, n_segments), rng.normal(27, 2, n_segments))
    tin = rng.normal(32, 4, n_segments)
    tout = tin - np.clip(rng.normal(3, 2, n_segments), 0.5, 10)
    pressure = rng.normal(55, 6, n_segments)
    flow = np.clip(rng.normal(2, 0.6, n_segments), 0.3, 4)
    shear = 0.3 * flow / (pressure / 50)
    wc = rng.beta(2, 10, n_segments)
    inhib = (rng.random(n_segments) < 0.3).astype(float)
    return wat, tin, tout, wc, inhib, shear, hw

def logistic_risk_probability(wat, tin, tout, wc, inhib, shear, hw):
    avg = 0.5 * (tin + tout)
    margin = wat - avg
    logit = 0.8 * (margin > 0) + 0.4 * (margin > 2) + 0.3 * (wc > 0.2) + 0.3 * (hw > 0.5)
    logit -= 0.4 * inhib + 0.3 * (shear > 0.9) + 0.4
    return 1 / (1 + np.exp(-logit))
