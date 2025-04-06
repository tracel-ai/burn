use backend_comparison::persistence::save;
use burn::backend::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::{
    BalancedCheckpointing, CheckpointStrategy, NoCheckpointing,
};
use burn::tensor::{Distribution, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, run_benchmark};
use core::f64::consts::SQRT_2;
use derive_new::new;

#[derive(Debug)]
enum GeluKind {
    Reference,
    WithReferenceErf,
    WithCustomErf,
}

/// Benchmark how well a backend executes a custom activation function with a lot of basic tensor
/// operations.
#[derive(new)]
struct CustomGeluBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
    kind: GeluKind,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Autodiff { gradient_checkpointing: bool },
    Inference,
}

impl core::fmt::Debug for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Autodiff {
                gradient_checkpointing,
            } => {
                if *gradient_checkpointing {
                    f.write_str("autodiff-checkpointing")
                } else {
                    f.write_str("autodiff")
                }
            }
            Mode::Inference => Ok(()),
        }
    }
}

impl<B: Backend, const D: usize> CustomGeluBenchmark<B, D> {
    fn execute_autodiff<C: CheckpointStrategy>(&self, tensor: Tensor<B, D>) {
        let tensor: Tensor<Autodiff<B, C>, D> = Tensor::from_inner(tensor).require_grad();
        let output = match self.kind {
            GeluKind::Reference => burn::tensor::activation::gelu(tensor.clone()),
            GeluKind::WithReferenceErf => gelu_custom(tensor.clone(), Tensor::erf),
            GeluKind::WithCustomErf => gelu_custom(tensor.clone(), erf_custom),
        };
        let mut gradients = output.sum().backward();
        let _tmp = tensor.grad_remove(&mut gradients).unwrap();
    }
}

impl<B: Backend, const D: usize> Benchmark for CustomGeluBenchmark<B, D> {
    type Args = Tensor<B, D>;

    fn name(&self) -> String {
        format!("gelu-{:?}-{:?}", self.kind, self.mode).to_lowercase()
    }

    fn options(&self) -> Option<String> {
        Some(format!("{:?}", self.kind))
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, tensor: Self::Args) {
        match self.mode {
            Mode::Autodiff {
                gradient_checkpointing,
            } => {
                if gradient_checkpointing {
                    self.execute_autodiff::<BalancedCheckpointing>(tensor)
                } else {
                    self.execute_autodiff::<NoCheckpointing>(tensor)
                }
            }
            Mode::Inference => {
                match self.kind {
                    GeluKind::Reference => burn::tensor::activation::gelu(tensor),
                    GeluKind::WithReferenceErf => gelu_custom(tensor, Tensor::erf),
                    GeluKind::WithCustomErf => gelu_custom(tensor, erf_custom),
                };
            }
        }
    }

    fn prepare(&self) -> Self::Args {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }

    fn num_samples(&self) -> usize {
        10
    }
}

fn gelu_custom<B, const D: usize, Erf>(x: Tensor<B, D>, erf: Erf) -> Tensor<B, D>
where
    B: Backend,
    Erf: Fn(Tensor<B, D>) -> Tensor<B, D>,
{
    let x = x.clone() * (erf(x / SQRT_2) + 1);
    x / 2
}

fn erf_custom<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let x1 = -erf_positive(-x.clone());
    let x2 = erf_positive(x.clone());
    let mask = x.greater_elem(0);

    x1.mask_where(mask, x2)
}

/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
fn erf_positive<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let x1 = x.clone().abs() * p + 1;
    let t = x1.recip();
    let tmp = (((((t.clone() * a5) + a4) * t.clone()) + a3) * t.clone() + a2) * t.clone() + a1;

    -(tmp * t * (-x.clone() * x).exp()) + 1.0
}

#[allow(dead_code)]
fn bench<B: Backend>(
    device: &B::Device,
    feature_name: &str,
    url: Option<&str>,
    token: Option<&str>,
) {
    const D: usize = 3;
    let shape: Shape = [32, 512, 2048].into();

    let run = |mode: Mode| {
        let reference_gelu = CustomGeluBenchmark::<B, D>::new(
            shape.clone(),
            device.clone(),
            GeluKind::Reference,
            mode,
        );
        let reference_erf_gelu = CustomGeluBenchmark::<B, D>::new(
            shape.clone(),
            device.clone(),
            GeluKind::WithReferenceErf,
            mode,
        );
        let custom_erf_gelu = CustomGeluBenchmark::<B, D>::new(
            shape.clone(),
            device.clone(),
            GeluKind::WithCustomErf,
            mode,
        );

        save::<B>(
            vec![
                run_benchmark(reference_gelu),
                run_benchmark(reference_erf_gelu),
                run_benchmark(custom_erf_gelu),
            ],
            device,
            feature_name,
            url,
            token,
        )
        .unwrap();
    };
    run(Mode::Inference);
    run(Mode::Autodiff {
        gradient_checkpointing: false,
    });
    run(Mode::Autodiff {
        gradient_checkpointing: true,
    });
}

fn main() {
    backend_comparison::bench_on_backend!();
}
