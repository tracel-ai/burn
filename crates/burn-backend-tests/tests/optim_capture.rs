//! Graph-captured optimizer steps must train exactly like eager steps.
//!
//! A replay runs the recorded kernels against the buffers the recording read and wrote, not the
//! closure. So an optimizer state (a momentum, a moment, the parameter itself) only advances
//! across replays when every update lands in the buffer it was read from, a property of the whole
//! step rather than of any one op: it breaks silently, e.g. when a `clone` gives a state tensor a
//! second owner and its update moves to a fresh buffer.
//!
//! Each test is black box: it captures a full training step (forward, backward, optimizer) of a
//! small model, replays it, and checks the losses and the parameters left in the model against the
//! same steps run eagerly. Every optimizer runs twice: with a constant host learning rate, and with
//! a decaying device learning rate written before each step, as a scheduler would. To cover a new
//! optimizer, add it to the list at the bottom.
//!
//! Isolated in this test binary for the same reason as `graph_capture`: a capture arms
//! device-global allocation state, so the tests are `#[serial]` and share no device with others.

#![cfg(feature = "cube")]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

use burn_core as burn;

use burn::module::{Module, ParamGroup};
use burn_nn::{Linear, LinearConfig};
use burn_optim::decay::WeightDecayConfig;
use burn_optim::momentum::MomentumConfig;
use burn_optim::{
    AdaGradConfig, AdafactorConfig, AdamConfig, AdamWConfig, AdanConfig, GradientsParams,
    LambConfig, LionConfig, ModuleOptimizer, MuonConfig, RmsPropConfig, SgdConfig,
};
use burn_tensor::{Device, Distribution, Tolerance, activation::relu};
use serial_test::serial;
use std::cell::{Cell, RefCell};

const LR: f64 = 1e-2;
/// The decay of [`Schedule::DecayingDevice`], per step.
const DECAY: f64 = 0.9;
/// Eager steps before capturing, so the optimizer state exists when the step is recorded.
const STEPS_BEFORE_CAPTURE: usize = 2;
const REPLAYS: usize = 5;
/// Replays run the same kernels as eager steps, so they should agree to rounding.
const MAX_DIFF: FloatElem = 1e-6;

/// The learning rate a test trains with.
#[derive(Clone, Copy, Debug)]
enum Schedule {
    /// [`LR`] on every step, as an `f64`.
    ConstantHost,
    /// [`LR`] decayed by [`DECAY`] per step, in a device tensor written before each step.
    DecayingDevice,
}

impl Schedule {
    /// The learning rate of the `step`-th step.
    fn lr(self, step: usize) -> f64 {
        match self {
            Schedule::ConstantHost => LR,
            Schedule::DecayingDevice => LR * DECAY.powi(step as i32),
        }
    }
}

#[derive(Module, Debug)]
struct Mlp {
    l1: Linear,
    l2: Linear,
}

/// A training step on a fixed batch.
struct Trainer {
    model: Option<Mlp>,
    optim: ModuleOptimizer,
    schedule: Schedule,
    /// The host learning rate of the next step.
    lr: f64,
    /// The learning rate buffer a device schedule writes into; a captured step reads it there.
    lr_device: Tensor<1>,
    x: Tensor<2>,
    y: Tensor<2>,
}

impl Trainer {
    fn new(optim: ModuleOptimizer, schedule: Schedule) -> Self {
        let device = Device::default();
        device.seed(0);
        let autodiff = device.clone().autodiff();
        let model = Mlp {
            l1: LinearConfig::new(16, 32).init(&autodiff),
            l2: LinearConfig::new(32, 4).init(&autodiff),
        };
        let x = Tensor::random([8, 16], Distribution::Normal(0.0, 1.0), &device);
        let y = Tensor::random([8, 4], Distribution::Normal(0.0, 1.0), &device);

        Self {
            model: Some(model),
            optim,
            schedule,
            lr: schedule.lr(0),
            lr_device: Tensor::from_floats([schedule.lr(0)], &device),
            x,
            y,
        }
    }

    /// Set the learning rate of the `step`-th step, outside any captured closure.
    fn set_lr(&mut self, step: usize) {
        self.lr = self.schedule.lr(step);
        if let Schedule::DecayingDevice = self.schedule {
            let lr = self.lr;
            // An input of the graph is refreshed in its own buffer.
            self.lr_device
                .inplace(|tensor| tensor.mul_scalar(0.0).add_scalar(lr));
        }
    }

    /// One step; the loss before the update.
    fn step(&mut self) -> Tensor<1> {
        let model = self.model.take().unwrap();
        let x = self.x.clone().autodiff();
        let pred = model.l2.forward(relu(model.l1.forward(x)));
        let loss = (pred - self.y.clone().autodiff()).square().mean();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        self.model = Some(match self.schedule {
            Schedule::ConstantHost => self.optim.step(self.lr, model, grads),
            Schedule::DecayingDevice => self.optim.step(self.lr_device.clone(), model, grads),
        });
        loss.inner()
    }

    fn params(&self) -> Vec<burn_tensor::TensorData> {
        let model = self.model.as_ref().unwrap();
        [&model.l1, &model.l2]
            .into_iter()
            .flat_map(|layer| {
                let bias = layer.bias.as_ref().map(|bias| bias.val().into_data());
                [Some(layer.weight.val().into_data()), bias]
            })
            .flatten()
            .collect()
    }
}

/// Replays of a captured training step match the same steps run eagerly.
///
/// Every closure run while capturing is a step at the learning rate set before `capture`; each
/// replay is one more step, at the learning rate set before it.
fn assert_capture_trains_like_eager(optim: fn() -> ModuleOptimizer, schedule: Schedule) {
    let device = Device::default();

    let captured = RefCell::new(Trainer::new(optim(), schedule));
    for step in 0..STEPS_BEFORE_CAPTURE {
        let mut trainer = captured.borrow_mut();
        trainer.set_lr(step);
        trainer.step();
    }
    captured.borrow_mut().set_lr(STEPS_BEFORE_CAPTURE);
    let runs = Cell::new(0);
    let mut graph = burn_tensor::capture(&device, || {
        runs.set(runs.get() + 1);
        captured.borrow_mut().step()
    });
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    assert!(graph.is_hardware(), "the training step should be captured");
    // Every closure run was a real step, except the recording of a hardware graph.
    let steps_during_capture = runs.get() - graph.is_hardware() as usize;

    // Safety: the graph's tensors are owned by `captured`, which outlives the graph, and every
    // replay and read happens sequentially on this thread.
    let first_replay = STEPS_BEFORE_CAPTURE + steps_during_capture;
    let replayed: Vec<FloatElem> = (0..REPLAYS)
        .map(|replay| {
            captured.borrow_mut().set_lr(first_replay + replay);
            unsafe { graph.replay() }.clone().into_scalar()
        })
        .collect();

    let mut eager = Trainer::new(optim(), schedule);
    for step in 0..first_replay {
        // The steps run while capturing all share the learning rate set before `capture`.
        eager.set_lr(step.min(STEPS_BEFORE_CAPTURE));
        eager.step();
    }
    for (step, replayed) in replayed.into_iter().enumerate() {
        eager.set_lr(first_replay + step);
        let expected = eager.step().into_scalar::<FloatElem>();
        assert!(
            (replayed - expected).abs() <= MAX_DIFF,
            "replay {step}: loss {replayed} while the eager step gives {expected}"
        );
    }

    // The model the caller holds must carry the replayed updates, not only the loss.
    let params = captured.borrow().params();
    for (param, expected) in params.iter().zip(eager.params()) {
        param.assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(MAX_DIFF));
    }
}

macro_rules! capture_trains_like_eager {
    ($($(#[$attr:meta])* $name:ident => $optim:expr,)*) => {
        $(
            mod $name {
                use super::*;

                #[test]
                #[serial]
                $(#[$attr])*
                fn constant_host_lr() {
                    assert_capture_trains_like_eager(|| $optim, Schedule::ConstantHost);
                }

                #[test]
                #[serial]
                $(#[$attr])*
                fn decaying_device_lr() {
                    assert_capture_trains_like_eager(|| $optim, Schedule::DecayingDevice);
                }
            }
        )*
    };
}

// The optimizers to validate. Add a new stateful optimizer (or configuration) here.
capture_trains_like_eager! {
    sgd => SgdConfig::new().init(),
    sgd_momentum => SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new()))
        .init(),
    sgd_nesterov_weight_decay => SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_nesterov(true)))
        .with_weight_decay(Some(WeightDecayConfig::new(0.01)))
        .init(),
    rmsprop_centered_momentum => RmsPropConfig::new()
        .with_centered(true)
        .with_momentum(0.9)
        .init(),
    adagrad => AdaGradConfig::new().init(),
    lion => LionConfig::new().with_weight_decay(0.1).init(),
    // Muon only takes matrices: the biases fall back to SGD with momentum.
    muon => SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new()))
        .init()
        .with_group(
            ParamGroup::from_regex(r"weight$").unwrap(),
            MuonConfig::new().build(),
            None,
        ),
    #[ignore = "the bias correction reads a host step counter, frozen at capture (#5779)"]
    adam => AdamConfig::new().init(),
    #[ignore = "the bias correction reads a host step counter, frozen at capture (#5779)"]
    adamw => AdamWConfig::new().init(),
    #[ignore = "the bias correction reads a host step counter, frozen at capture (#5779)"]
    lamb => LambConfig::new().init(),
    #[ignore = "the moment weights read a host step counter, frozen at capture (#5779)"]
    adafactor => AdafactorConfig::new().init(),
    #[ignore = "a host step counter, and `neg_pre_grad` lands in the gradient's buffer (#5779)"]
    adan => AdanConfig::new().init(),
}
