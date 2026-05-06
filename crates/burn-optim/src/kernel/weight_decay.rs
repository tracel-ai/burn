use cubecl::prelude::*;

#[cube]
pub fn weight_decay(
    i: u32,
    tensor: &Array<f32>,
    raw_delta: &Array<f32>,
    m1_dequantized: &Array<f32>,
    tensor_out: &mut Array<f32>,

    lr: f32,
    decay_rate: f32,

    #[comptime] cautious_weight_decay: bool,
) {
    let i = i as usize;

    let theta = tensor[i];
    let delta = raw_delta[i];

    // Branch on whether decay applies to this element.
    // - If decay_rate is exactly 0.0, decay is a no-op (skip multiply).
    // - If cautious, only decay when sign(theta) matches sign(m1).
    // - Otherwise, plain decay.
    let decayed = if decay_rate == 0.0f32 {
        theta
    } else if comptime!(cautious_weight_decay) {
        let m1 = m1_dequantized[i];
        let theta_pos = theta >= 0.0f32;
        let m1_pos = m1 >= 0.0f32;
        let sign_agrees = theta_pos == m1_pos;
        if sign_agrees {
            theta - theta * decay_rate
        } else {
            theta
        }
    } else {
        theta * (1.0f32 - decay_rate)
    };

    tensor_out[i] = decayed - lr * delta;
}
