//! CPU timings for softmax, log-softmax, and hard-target cross-entropy.
//!
//! Run from the repository root:
//! `cargo bench -p burn-nn --locked --features burn-core/flex --bench softmax_collapse`
//! Set `BURN_BENCH_DTYPE=f16` or `BURN_BENCH_DTYPE=bf16` for half-precision timings.

use burn_core::tensor::{DType, Device, DeviceConfig, Int, Tensor, TensorData, activation};
use burn_nn::loss::CrossEntropyLossConfig;
use std::{hint::black_box, time::Instant};

fn median_microseconds<O>(device: &Device, operation: impl Fn() -> O) -> f64 {
    const ITERATIONS: usize = 20;
    const SAMPLES: usize = 7;

    for _ in 0..5 {
        let output = black_box(operation());
        device.sync().unwrap();
        drop(output);
    }

    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        device.sync().unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let output = black_box(operation());
            device.sync().unwrap();
            drop(output);
        }
        *sample = start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[SAMPLES / 2]
}

fn main() {
    let dtype = match std::env::var("BURN_BENCH_DTYPE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("f32") => DType::F32,
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        _ => panic!("BURN_BENCH_DTYPE must be f32, f16, or bf16"),
    };
    let mut device = Device::flex();
    device
        .configure(
            DeviceConfig::default()
                .float_dtype(dtype)
                .int_dtype(DType::I32),
        )
        .unwrap();
    let autodiff = device.clone().autodiff();
    println!("Flex CPU, {dtype:?}, dim=1, median of 7 samples x 20 iterations; 5 warm-ups");
    println!("shape,distribution,operation,forward_us,forward_backward_us");

    for classes in [3, 2048] {
        for high_confidence in [false, true] {
            let shape = [32, classes];
            let values: Vec<f32> = (0..32 * classes)
                .map(|index| {
                    if high_confidence {
                        if index % classes == 0 { 20.0 } else { -20.0 }
                    } else {
                        (index % 97) as f32 / 48.0 - 1.0
                    }
                })
                .collect();
            let data = TensorData::new(values, shape);
            let input = Tensor::<2>::from_data(data.clone(), &device);
            let tracked = Tensor::<2>::from_data(data, &autodiff).require_grad();
            assert_eq!(input.dtype(), dtype);
            assert_eq!(tracked.dtype(), dtype);
            let targets = Tensor::<1, Int>::zeros([32], &device);
            let tracked_targets = Tensor::<1, Int>::zeros([32], &autodiff);
            let loss = CrossEntropyLossConfig::new().init(&device);
            let tracked_loss = CrossEntropyLossConfig::new().init(&autodiff);
            let distribution = if high_confidence {
                "high_confidence_gap40"
            } else {
                "ordinary"
            };

            let forward = median_microseconds(&device, || activation::softmax(input.clone(), 1));
            let backward = median_microseconds(&autodiff, || {
                let output = activation::softmax(tracked.clone(), 1).slice([0..32, 0..1]);
                let grads = output.sum().backward();
                tracked.grad(&grads).unwrap()
            });
            println!("32x{classes},{distribution},softmax,{forward:.3},{backward:.3}");

            let forward =
                median_microseconds(&device, || activation::log_softmax(input.clone(), 1));
            let backward = median_microseconds(&autodiff, || {
                let output = activation::log_softmax(tracked.clone(), 1).slice([0..32, 0..1]);
                let grads = output.sum().backward();
                tracked.grad(&grads).unwrap()
            });
            println!("32x{classes},{distribution},log_softmax,{forward:.3},{backward:.3}");

            let forward =
                median_microseconds(&device, || loss.forward(input.clone(), targets.clone()));
            let backward = median_microseconds(&autodiff, || {
                let output = tracked_loss.forward(tracked.clone(), tracked_targets.clone());
                let grads = output.backward();
                tracked.grad(&grads).unwrap()
            });
            println!("32x{classes},{distribution},cross_entropy,{forward:.3},{backward:.3}");
        }
    }
}
