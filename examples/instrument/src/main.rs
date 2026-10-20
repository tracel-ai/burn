//! Collect one pass of a small model and save it for `burn-instrument-report`.
//!
//! ```sh
//! cargo run --release -p instrument --features rocm -- pass.json
//! cargo run --release -p burn-instrument-report -- pass.json
//! ```
//!
//! The device is `BURN_DEVICE`'s, or the default one of the linked runtime.

use burn::instrument::{Bounds, Instrument, compute, tensor_bytes};
use burn::tensor::activation::gelu;
use burn::tensor::{Device, Distribution, FloatDType, Tensor};

/// Rows of the input, as a batch of tokens.
const TOKENS: usize = 512;
/// Width of the residual stream.
const WIDTH: usize = 1024;
/// Width of the feed-forward's hidden layer.
const HIDDEN: usize = 4 * WIDTH;
/// Blocks in the stack.
const LAYERS: usize = 4;
/// Passes run before anything is measured: the first compiles and autotunes.
const WARMUP: usize = 5;
/// Uncollected passes timed for the clean wall, of which the median is kept.
const SAMPLES: usize = 10;

/// A weight multiplied into the residual stream, in one of several roles under its block.
struct Projection {
    /// Which projection this is: two `Projection`s under one block are two components only
    /// if they are named apart.
    role: &'static str,
    weight: Tensor<2>,
}

impl Projection {
    fn new(role: &'static str, input: usize, output: usize, device: &Device) -> Self {
        let scale = (1.0 / input as f64).sqrt();
        Self {
            role,
            weight: Tensor::random(
                [input, output],
                Distribution::Uniform(-scale, scale),
                device,
            ),
        }
    }

    #[compute(label = self.role, bounds = self.bounds(&x))]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.clone())
    }

    /// Read the input and the weight once, write the output once.
    fn bounds(&self, x: &Tensor<2>) -> Bounds {
        let [rows, input] = x.dims();
        let [_, output] = self.weight.dims();
        let element = tensor_bytes(x) / (rows * input) as u64;
        Bounds {
            read: tensor_bytes(x) + tensor_bytes(&self.weight),
            write: (rows * output) as u64 * element,
            flops: 2 * (rows * input * output) as u64,
        }
    }
}

/// A residual feed-forward block: `x + down(gelu(up(norm(x))))`.
struct Block {
    up: Projection,
    down: Projection,
}

impl Block {
    fn new(device: &Device) -> Self {
        Self {
            up: Projection::new("up", WIDTH, HIDDEN, device),
            down: Projection::new("down", HIDDEN, WIDTH, device),
        }
    }

    /// Unpriced: the norm, the activation and the residual add are left for the fuser to fold
    /// into the projections, so the block's floor is its projections'.
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let hidden = gelu(self.up.forward(Norm.forward(x.clone())));
        x + self.down.forward(hidden)
    }
}

/// Scale each row to unit root-mean-square.
struct Norm;

impl Norm {
    #[compute]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let rms = (x.clone().powf_scalar(2.0).mean_dim(1) + 1e-6).sqrt();
        x / rms
    }
}

/// The stack.
struct Model {
    blocks: Vec<Block>,
}

impl Model {
    #[compute(bounds = self.bounds(&x))]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.blocks.iter().fold(x, |x, block| block.forward(x))
    }

    /// Read the input and every weight once, write the output once: what the whole stack
    /// moves with nothing of its own reaching memory twice.
    fn bounds(&self, x: &Tensor<2>) -> Bounds {
        let weights = self
            .blocks
            .iter()
            .flat_map(|block| [&block.up, &block.down])
            .map(|projection| tensor_bytes(&projection.weight))
            .sum::<u64>();
        let flops = 2 * 2 * (TOKENS * WIDTH * HIDDEN * LAYERS) as u64;
        Bounds {
            read: tensor_bytes(x) + weights,
            write: tensor_bytes(x),
            flops,
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pass.json".to_string());
    let device = Device::default();
    let model = Model {
        blocks: (0..LAYERS).map(|_| Block::new(&device)).collect(),
    };
    let input = Tensor::<2>::random([TOKENS, WIDTH], Distribution::Default, &device);

    let measurement = Instrument::new(&device)
        .warmup(WARMUP)
        .samples(SAMPLES)
        .dtype(FloatDType::F32)
        .measure(|| model.forward(input.clone()));
    let file = std::fs::File::create(&path).expect("the pass file can be created");
    measurement.write(file).expect("the pass is written");
    println!("saved {path}");
}
