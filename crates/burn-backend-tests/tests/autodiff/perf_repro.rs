//! Ad-hoc reproducer for the superlinear forward+backward scaling reported against an unrolled
//! per-timestep recurrence (5 "layers" x T timesteps x sigmoid/tanh/mul/add gates, matching a
//! BiLSTM-shaped autodiff graph). Only meaningful under the `checkpointing` variant of this
//! module (gradient checkpointing enabled), which is the path that exercises
//! `Checkpointer::topological_sort` in `crates/burn-autodiff/src/checkpoint/base.rs`.
//!
//! Run with:
//! `cargo test -p burn-backend-tests --test autodiff -- checkpointing::perf_repro --nocapture --test-threads=1`
use super::*;
use std::time::Instant;

fn lstm_like_forward_backward(steps: usize) -> f64 {
    let device = AutodiffDevice::new();
    let batch = 16;
    let input = 4;
    let hidden = 64;
    let layers = 5;

    let start = Instant::now();

    let mut layer_in_dim = input;
    let mut inputs: Vec<TestTensor<2>> = (0..steps)
        .map(|_| {
            TestTensor::<2>::random(
                [batch, layer_in_dim],
                burn_tensor::Distribution::Default,
                &device,
            )
        })
        .collect();

    for _layer in 0..layers {
        let w_i = TestTensor::<2>::random(
            [layer_in_dim, hidden],
            burn_tensor::Distribution::Default,
            &device,
        )
        .require_grad();
        let w_h = TestTensor::<2>::random(
            [hidden, hidden],
            burn_tensor::Distribution::Default,
            &device,
        )
        .require_grad();

        let mut h = TestTensor::<2>::zeros([batch, hidden], &device).require_grad();
        let mut c = TestTensor::<2>::zeros([batch, hidden], &device).require_grad();

        let mut outputs = Vec::with_capacity(steps);
        for x_t in inputs.iter() {
            let gate_in = x_t.clone().matmul(w_i.clone()) + h.clone().matmul(w_h.clone());
            let i_gate = burn_tensor::activation::sigmoid(gate_in.clone());
            let f_gate = burn_tensor::activation::sigmoid(gate_in.clone());
            let g_gate = burn_tensor::activation::tanh(gate_in.clone());
            let o_gate = burn_tensor::activation::sigmoid(gate_in);
            c = f_gate * c + i_gate * g_gate;
            h = o_gate * burn_tensor::activation::tanh(c.clone());
            outputs.push(h.clone());
        }

        inputs = outputs;
        layer_in_dim = hidden;
    }

    let mut loss = inputs[0].clone().sum();
    for t in inputs.iter().skip(1) {
        loss = loss + t.clone().sum();
    }

    let _grads = loss.backward();

    start.elapsed().as_secs_f64()
}

#[test]
fn perf_repro_020() {
    let secs = lstm_like_forward_backward(20);
    println!("perf_repro steps=20   -> {secs:.3}s");
}

#[test]
fn perf_repro_080() {
    let secs = lstm_like_forward_backward(80);
    println!("perf_repro steps=80   -> {secs:.3}s");
}

#[test]
fn perf_repro_320() {
    let secs = lstm_like_forward_backward(320);
    println!("perf_repro steps=320  -> {secs:.3}s");
}
