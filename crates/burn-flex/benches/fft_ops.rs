//! Benchmarks comparing Flex rfft vs realfft (rustfft-backed) crate.
//!
//! Run with:
//! ```bash
//! cargo bench --bench fft_ops --features simd
//! ```

use burn_flex::Flex;
use burn_tensor::{
    Tensor, TensorData,
    signal::{irfft, rfft},
};
use divan::{AllocProfiler, Bencher};
use realfft::RealFftPlanner;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Flex vs realfft (rustfft) benchmarks");
    println!();
    divan::main();
}

type B = Flex;

fn make_signal_1d(n: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    Tensor::from_data(TensorData::new(data, [n]), &Default::default())
}

fn make_signal_2d(batch: usize, n: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..batch * n).map(|i| (i as f32 * 0.1).sin()).collect();
    Tensor::from_data(TensorData::new(data, [batch, n]), &Default::default())
}

fn make_raw_signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.1).sin()).collect()
}

// ============================================================================
// Flex backend
// ============================================================================

#[divan::bench_group(name = "flex")]
mod flex {
    use super::*;

    #[divan::bench_group(name = "rfft_1d")]
    mod rfft_1d {
        use super::*;

        #[divan::bench]
        fn n_256(bencher: Bencher) {
            let s = make_signal_1d(256);
            bencher.bench(|| rfft(s.clone(), 0));
        }

        #[divan::bench]
        fn n_1024(bencher: Bencher) {
            let s = make_signal_1d(1024);
            bencher.bench(|| rfft(s.clone(), 0));
        }

        #[divan::bench]
        fn n_4096(bencher: Bencher) {
            let s = make_signal_1d(4096);
            bencher.bench(|| rfft(s.clone(), 0));
        }

        #[divan::bench]
        fn n_16384(bencher: Bencher) {
            let s = make_signal_1d(16384);
            bencher.bench(|| rfft(s.clone(), 0));
        }

        #[divan::bench]
        fn n_65536(bencher: Bencher) {
            let s = make_signal_1d(65536);
            bencher.bench(|| rfft(s.clone(), 0));
        }
    }

    #[divan::bench_group(name = "rfft_2d_batch")]
    mod rfft_2d_batch {
        use super::*;

        #[divan::bench]
        fn batch_16_n_1024(bencher: Bencher) {
            let s = make_signal_2d(16, 1024);
            bencher.bench(|| rfft(s.clone(), 1));
        }

        #[divan::bench]
        fn batch_64_n_1024(bencher: Bencher) {
            let s = make_signal_2d(64, 1024);
            bencher.bench(|| rfft(s.clone(), 1));
        }

        #[divan::bench]
        fn batch_256_n_256(bencher: Bencher) {
            let s = make_signal_2d(256, 256);
            bencher.bench(|| rfft(s.clone(), 1));
        }
    }

    #[divan::bench_group(name = "irfft_1d")]
    mod irfft_1d {
        use super::*;

        fn bench_irfft(bencher: Bencher, n: usize) {
            let s = make_signal_1d(n);
            let (re, im) = rfft(s, 0);
            bencher.bench(|| irfft(re.clone(), im.clone(), 0));
        }

        #[divan::bench]
        fn n_256(bencher: Bencher) {
            bench_irfft(bencher, 256);
        }

        #[divan::bench]
        fn n_1024(bencher: Bencher) {
            bench_irfft(bencher, 1024);
        }

        #[divan::bench]
        fn n_4096(bencher: Bencher) {
            bench_irfft(bencher, 4096);
        }

        #[divan::bench]
        fn n_16384(bencher: Bencher) {
            bench_irfft(bencher, 16384);
        }

        #[divan::bench]
        fn n_65536(bencher: Bencher) {
            bench_irfft(bencher, 65536);
        }
    }
}

// ============================================================================
// realfft (rustfft-backed)
// ============================================================================

#[divan::bench_group(name = "realfft")]
mod realfft_bench {
    use super::*;

    #[divan::bench_group(name = "rfft_1d")]
    mod rfft_1d {
        use super::*;

        fn bench_realfft(bencher: Bencher, n: usize) {
            let mut planner = RealFftPlanner::<f32>::new();
            let r2c = planner.plan_fft_forward(n);
            let signal = make_raw_signal(n);
            bencher.bench(|| {
                let mut input = signal.clone();
                let mut spectrum = r2c.make_output_vec();
                r2c.process(&mut input, &mut spectrum).unwrap();
                spectrum
            });
        }

        #[divan::bench]
        fn n_256(bencher: Bencher) {
            bench_realfft(bencher, 256);
        }

        #[divan::bench]
        fn n_1024(bencher: Bencher) {
            bench_realfft(bencher, 1024);
        }

        #[divan::bench]
        fn n_4096(bencher: Bencher) {
            bench_realfft(bencher, 4096);
        }

        #[divan::bench]
        fn n_16384(bencher: Bencher) {
            bench_realfft(bencher, 16384);
        }

        #[divan::bench]
        fn n_65536(bencher: Bencher) {
            bench_realfft(bencher, 65536);
        }
    }

    #[divan::bench_group(name = "rfft_2d_batch")]
    mod rfft_2d_batch {
        use super::*;

        fn bench_realfft_batch(bencher: Bencher, batch: usize, n: usize) {
            let mut planner = RealFftPlanner::<f32>::new();
            let r2c = planner.plan_fft_forward(n);
            let signal = make_raw_signal(batch * n);
            bencher.bench(|| {
                let mut spectra = Vec::with_capacity(batch);
                for b in 0..batch {
                    let mut input = signal[b * n..(b + 1) * n].to_vec();
                    let mut spectrum = r2c.make_output_vec();
                    r2c.process(&mut input, &mut spectrum).unwrap();
                    spectra.push(spectrum);
                }
                spectra
            });
        }

        #[divan::bench]
        fn batch_16_n_1024(bencher: Bencher) {
            bench_realfft_batch(bencher, 16, 1024);
        }

        #[divan::bench]
        fn batch_64_n_1024(bencher: Bencher) {
            bench_realfft_batch(bencher, 64, 1024);
        }

        #[divan::bench]
        fn batch_256_n_256(bencher: Bencher) {
            bench_realfft_batch(bencher, 256, 256);
        }
    }

    #[divan::bench_group(name = "irfft_1d")]
    mod irfft_1d {
        use super::*;

        fn bench_realfft_inverse(bencher: Bencher, n: usize) {
            let mut planner = RealFftPlanner::<f32>::new();
            let r2c = planner.plan_fft_forward(n);
            let c2r = planner.plan_fft_inverse(n);
            let signal = make_raw_signal(n);
            let mut input = signal.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();
            bencher.bench(|| {
                let mut spec = spectrum.clone();
                let mut output = c2r.make_output_vec();
                c2r.process(&mut spec, &mut output).unwrap();
                output
            });
        }

        #[divan::bench]
        fn n_256(bencher: Bencher) {
            bench_realfft_inverse(bencher, 256);
        }

        #[divan::bench]
        fn n_1024(bencher: Bencher) {
            bench_realfft_inverse(bencher, 1024);
        }

        #[divan::bench]
        fn n_4096(bencher: Bencher) {
            bench_realfft_inverse(bencher, 4096);
        }

        #[divan::bench]
        fn n_16384(bencher: Bencher) {
            bench_realfft_inverse(bencher, 16384);
        }

        #[divan::bench]
        fn n_65536(bencher: Bencher) {
            bench_realfft_inverse(bencher, 65536);
        }
    }
}
