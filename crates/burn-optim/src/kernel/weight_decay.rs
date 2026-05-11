use cubecl::prelude::*;

#[cube]
pub fn weight_decay(
    i: u32,
    iter: u32,
    theta: &mut Array<f32>,
    local_delta: &Array<f32>,
    local_m1: &Array<f32>,
    lr: f32,
    decay_rate: f32,
    #[comptime] cautious_weight_decay: bool,
) {
    let i = i as usize;
    let iter = iter as usize;
    let theta_val = theta[i];
    let delta = local_delta[iter];

    let decayed = if decay_rate == 0.0f32 {
        theta_val
    } else if comptime!(cautious_weight_decay) {
        let m1 = local_m1[iter];
        let theta_pos = theta_val >= 0.0f32;
        let m1_pos = m1 >= 0.0f32;
        let sign_agrees = theta_pos == m1_pos;
        if sign_agrees {
            theta_val - theta_val * decay_rate
        } else {
            theta_val
        }
    } else {
        theta_val * (1.0f32 - decay_rate)
    };
    theta[i] = decayed - lr * delta;
}
