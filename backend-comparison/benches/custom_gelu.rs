use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};
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
    shape: Shape<D>,
    num_repeats: usize,
    device: B::Device,
    kind: GeluKind,
}

impl<B: Backend, const D: usize> Benchmark for CustomGeluBenchmark<B, D> {
    type Args = Tensor<B, D>;

    fn name(&self) -> String {
        format!("Gelu {:?}", self.kind)
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats {
            match self.kind {
                GeluKind::Reference => burn::tensor::activation::gelu(args.clone()),
                GeluKind::WithReferenceErf => gelu_custom(args.clone(), Tensor::erf),
                GeluKind::WithCustomErf => gelu_custom(args.clone(), erf_custom),
            };
        }
    }

    fn prepare(&self) -> Self::Args {
        Tensor::random_device(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

fn gelu_custom<B, const D: usize, Erf>(x: Tensor<B, D>, erf: Erf) -> Tensor<B, D>
where
    B: Backend,
    Erf: Fn(Tensor<B, D>) -> Tensor<B, D>,
{
    let x = x.clone() * (erf(x / SQRT_2) + 1);
    let result = x / 2;

    result
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

    return -(tmp * t * (-x.clone() * x).exp()) + 1.0;
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    const D: usize = 3;
    let shape: Shape<D> = [32, 512, 2048].into();
    let num_repeats = 1;

    println!("Backend {}", B::name());
    run_benchmark(CustomGeluBenchmark::<B, D>::new(
        shape.clone(),
        num_repeats,
        device.clone(),
        GeluKind::Reference,
    ));
    run_benchmark(CustomGeluBenchmark::<B, D>::new(
        shape.clone(),
        num_repeats,
        device.clone(),
        GeluKind::WithReferenceErf,
    ));
    run_benchmark(CustomGeluBenchmark::<B, D>::new(
        shape,
        num_repeats,
        device.clone(),
        GeluKind::WithCustomErf,
    ));
}

fn main() {
    backend_comparison::bench_on_backend!();
}
