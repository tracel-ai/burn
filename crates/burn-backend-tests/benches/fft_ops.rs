//! Benchmarks for rfft / irfft on TestBackend.
//!
//! Run with:
//! ```bash
//! cargo bench --bench fft_ops
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{
    Tensor, TensorData,
    signal::{irfft, rfft},
};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("rfft / irfft benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

type B = TestBackend;

fn make_signal_1d(n: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    Tensor::from_data(TensorData::new(data, [n]), &Default::default())
}

fn make_signal_2d(batch: usize, n: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..batch * n).map(|i| (i as f32 * 0.1).sin()).collect();
    Tensor::from_data(TensorData::new(data, [batch, n]), &Default::default())
}

#[divan::bench_group(name = "rfft_1d")]
mod rfft_1d {
    use super::*;

    #[divan::bench]
    fn n_256(bencher: Bencher) {
        let s = make_signal_1d(256);
        bencher.bench_synced(|| rfft(s.clone(), 0, None));
    }

    #[divan::bench]
    fn n_1024(bencher: Bencher) {
        let s = make_signal_1d(1024);
        bencher.bench_synced(|| rfft(s.clone(), 0, None));
    }

    #[divan::bench]
    fn n_4096(bencher: Bencher) {
        let s = make_signal_1d(4096);
        bencher.bench_synced(|| rfft(s.clone(), 0, None));
    }

    #[divan::bench]
    fn n_16384(bencher: Bencher) {
        let s = make_signal_1d(16384);
        bencher.bench_synced(|| rfft(s.clone(), 0, None));
    }

    #[divan::bench]
    fn n_65536(bencher: Bencher) {
        let s = make_signal_1d(65536);
        bencher.bench_synced(|| rfft(s.clone(), 0, None));
    }
}

#[divan::bench_group(name = "rfft_2d_batch")]
mod rfft_2d_batch {
    use super::*;

    #[divan::bench]
    fn batch_16_n_1024(bencher: Bencher) {
        let s = make_signal_2d(16, 1024);
        bencher.bench_synced(|| rfft(s.clone(), 1, None));
    }

    #[divan::bench]
    fn batch_64_n_1024(bencher: Bencher) {
        let s = make_signal_2d(64, 1024);
        bencher.bench_synced(|| rfft(s.clone(), 1, None));
    }

    #[divan::bench]
    fn batch_256_n_256(bencher: Bencher) {
        let s = make_signal_2d(256, 256);
        bencher.bench_synced(|| rfft(s.clone(), 1, None));
    }
}

#[divan::bench_group(name = "irfft_1d")]
mod irfft_1d {
    use super::*;

    fn bench_irfft(bencher: Bencher, n: usize) {
        // rfft is used to produce the input spectrum; if the backend doesn't implement it the
        // setup panics before bench_synced's catch_unwind. Use try_setup to record and fall
        // through to a no-op.
        let Some((re, im)) = common::try_setup(|| {
            let s = make_signal_1d(n);
            rfft(s, 0, None)
        }) else {
            bencher.bench(|| ());
            return;
        };
        bencher.bench_synced(|| irfft(re.clone(), im.clone(), 0, None));
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
