//! Runs a backend extension on the remote backend with no handwritten remote code: listing
//! `Remote` beside `Fusion` generates the client, which ships each call as a `CustomOpIr`, and the
//! server handlers, which rebuild the call and run it on the backend the server hosts.
//!
//! The server hosts a Flex device over a WebSocket loopback; every call is compared with the same
//! call on a local Flex device.
#![cfg(all(
    feature = "remote-server",
    feature = "remote-websocket",
    feature = "fusion",
    feature = "flex"
))]

use burn::backend::{
    Backend, Dispatch, ExtensionType, Flex, FloatDType, IntDType, backend_extension,
    fusion::custom::TensorSpec,
    ops::{BoolTensorOps, FloatTensorOps, IntTensorOps},
    tensor::{FloatTensor, IntTensor},
};
use burn::remote::RemoteHost;
use burn::server::{CustomOpRegistry, RemoteServer, WebSocketTransport};
use burn::tensor::{BoolDType, DType, Device, Int, Tensor};

/// How [`ExampleBackend::combine`] joins its two operands, sent as a code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Join {
    Add,
    Mul,
}

impl Join {
    fn to_code(self) -> u64 {
        match self {
            Join::Add => 0,
            Join::Mul => 1,
        }
    }

    fn from_code(code: u64) -> Option<Self> {
        match code {
            0 => Some(Join::Add),
            1 => Some(Join::Mul),
            _ => None,
        }
    }
}

/// An ordinary field with a custom scalar encoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factor(pub f32);

/// An enum input: the variant, an ordinary primitive field and a custom-encoded field all travel
/// as scalars.
#[derive(ExtensionType)]
#[extension_type(fusion, remote)]
pub enum Operand<B: Backend> {
    Dense {
        values: FloatTensor<B>,
    },
    Masked {
        values: FloatTensor<B>,
        mask: IntTensor<B>,
        fill: f32,
    },
    Scaled {
        values: FloatTensor<B>,
        #[extension_type(scalar = factor.0, decode = Factor)]
        factor: Factor,
    },
}

/// A struct output with a float tensor, an int tensor and an ordinary field.
#[derive(ExtensionType)]
#[extension_type(fusion, remote)]
pub struct Summary<B: Backend> {
    pub combined: FloatTensor<B>,
    pub positive: IntTensor<B>,
    pub count: usize,
}

#[backend_extension(Flex, Fusion, Remote)]
pub trait ExampleBackend: Backend {
    /// `x` joined with the operand, plus `bias`, which elements are positive, and how many
    /// elements there are.
    #[fusion(meta = |x, _bias, _join, _operand| SummaryMetadata {
        combined: x.clone(),
        positive: TensorSpec::new(x.shape.clone(), DType::I64),
        count: x.shape.num_elements(),
    })]
    fn combine(
        x: FloatTensor<Self>,
        bias: f32,
        #[fusion(scalar = join.to_code(), decode = |code| Join::from_code(code).expect("a join code"))]
        join: Join,
        #[extension_type] operand: Operand<Self>,
    ) -> Summary<Self>;

    /// The positions of the `k` largest elements of each row; an int output.
    #[fusion(dtype = DType::I64, shape = {
        let mut shape = scores.clone();
        *shape.last_mut().unwrap() = *k;
        shape
    })]
    fn top_positions(scores: FloatTensor<Self>, k: usize) -> IntTensor<Self>;
}

impl ExampleBackend for Flex {
    fn combine(
        x: FloatTensor<Self>,
        bias: f32,
        join: Join,
        operand: Operand<Self>,
    ) -> Summary<Self> {
        let count = burn::backend::TensorMetadata::shape(&x).num_elements();
        let other = match operand {
            Operand::Dense { values } => values,
            Operand::Masked { values, mask, fill } => {
                let mask = Flex::int_into_float(mask, FloatDType::F32);
                let kept = Flex::float_mul(values, mask.clone());
                let dropped = Flex::float_mul_scalar(
                    Flex::float_add_scalar(Flex::float_neg(mask), 1.0.into()),
                    fill.into(),
                );
                Flex::float_add(kept, dropped)
            }
            Operand::Scaled { values, factor } => Flex::float_mul_scalar(values, factor.0.into()),
        };
        let joined = match join {
            Join::Add => Flex::float_add(x, other),
            Join::Mul => Flex::float_mul(x, other),
        };
        let combined = Flex::float_add_scalar(joined, bias.into());
        let positive = Flex::bool_into_int(
            Flex::float_greater_elem(combined.clone(), 0.0.into(), BoolDType::Native),
            IntDType::I64,
        );
        Summary {
            combined,
            positive,
            count,
        }
    }

    fn top_positions(scores: FloatTensor<Self>, k: usize) -> IntTensor<Self> {
        let rank = burn::backend::TensorMetadata::shape(&scores).num_dims();
        let order = Flex::float_argsort(scores, rank - 1, true, IntDType::I64);
        let mut ranges = burn::backend::TensorMetadata::shape(&order)
            .iter()
            .map(|&dim| burn::tensor::Slice::from(0..dim))
            .collect::<Vec<_>>();
        ranges[rank - 1] = burn::tensor::Slice::from(0..k);
        Flex::int_slice(order, &ranges)
    }
}

/// Two traits whose methods share a name, served from one registry.
#[backend_extension(Flex, Fusion, Remote)]
pub trait ShiftBackend: Backend {
    /// `x` plus `amount`, negated when `negate` is set; a bool scalar.
    #[fusion(dtype = x, shape = x)]
    fn forward(x: FloatTensor<Self>, amount: f32, negate: bool) -> FloatTensor<Self>;
}

#[backend_extension(Flex, Fusion, Remote)]
pub trait RepeatBackend: Backend {
    /// `x` times `times`; a usize scalar.
    #[fusion(dtype = x, shape = x)]
    fn forward(x: FloatTensor<Self>, times: usize) -> FloatTensor<Self>;
}

/// Claims `ShiftBackend::forward`'s operation id with a different handler.
#[backend_extension(Flex, Fusion, Remote)]
pub trait ClashingBackend: Backend {
    #[fusion(id = "ShiftBackend::forward", dtype = x, shape = x)]
    fn clash(x: FloatTensor<Self>) -> FloatTensor<Self>;
}

impl ShiftBackend for Flex {
    fn forward(x: FloatTensor<Self>, amount: f32, negate: bool) -> FloatTensor<Self> {
        let shifted = Flex::float_add_scalar(x, amount.into());
        if negate {
            Flex::float_neg(shifted)
        } else {
            shifted
        }
    }
}

impl RepeatBackend for Flex {
    fn forward(x: FloatTensor<Self>, times: usize) -> FloatTensor<Self> {
        Flex::float_mul_scalar(x, (times as f32).into())
    }
}

impl ClashingBackend for Flex {
    fn clash(x: FloatTensor<Self>) -> FloatTensor<Self> {
        x
    }
}

/// Start a server hosting one Flex device with the extension's handlers, returning a remote
/// device on it.
fn remote_device() -> Device {
    let mut registry = CustomOpRegistry::<Flex>::new();
    ExampleBackendHandlers::register(&mut registry);
    serve(registry)
}

/// Start a server hosting one Flex device with `registry`, returning a remote device on it.
fn serve(registry: CustomOpRegistry<Flex>) -> Device {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()));
    let server = RemoteServer::new([Device::flex()]).with_custom_ops(registry);
    std::thread::spawn(move || server.serve(WebSocketTransport::from_listener(listener)));
    Device::remote_options(&host).init().unwrap()
}

/// What `combine` returns, read back to the host.
#[derive(Debug, PartialEq)]
struct Combined {
    combined: Vec<f32>,
    positive: Vec<i64>,
    count: usize,
}

/// The operands `combine` is called with, one per variant.
#[derive(Clone, Copy, Debug)]
enum Variant {
    Dense,
    Masked,
    Scaled,
}

fn combine(device: &Device, join: Join, variant: Variant) -> Combined {
    let x = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], device);
    let values = Tensor::<2>::from_floats([[0.5, 1.5, -2.5], [3.5, -4.5, 5.5]], device);
    let operand = match variant {
        Variant::Dense => Operand::Dense {
            values: values.into_dispatch(),
        },
        Variant::Masked => Operand::Masked {
            values: values.into_dispatch(),
            mask: Tensor::<2, Int>::from_ints([[1, 0, 1], [0, 1, 0]], device).into_dispatch(),
            fill: 7.0,
        },
        Variant::Scaled => Operand::Scaled {
            values: values.into_dispatch(),
            factor: Factor(-3.0),
        },
    };
    let summary = <Dispatch as ExampleBackend>::combine(x.into_dispatch(), 0.25, join, operand);
    Combined {
        combined: Tensor::<2>::from_dispatch(summary.combined)
            .into_data()
            .try_to_vec()
            .unwrap(),
        positive: Tensor::<2, Int>::from_dispatch(summary.positive)
            .into_data()
            .try_to_vec()
            .unwrap(),
        count: summary.count,
    }
}

fn top_positions(device: &Device, k: usize) -> Vec<i64> {
    let scores = Tensor::<2>::from_floats([[0.1, 0.9, 0.5, 0.7], [0.8, 0.2, 0.6, 0.4]], device);
    let positions = <Dispatch as ExampleBackend>::top_positions(scores.into_dispatch(), k);
    Tensor::<2, Int>::from_dispatch(positions)
        .into_data()
        .try_to_vec()
        .unwrap()
}

#[test]
fn generated_remote_extension_matches_local() {
    let remote = remote_device();
    let local = Device::flex();
    for join in [Join::Add, Join::Mul] {
        for variant in [Variant::Dense, Variant::Masked, Variant::Scaled] {
            assert_eq!(
                combine(&remote, join, variant),
                combine(&local, join, variant),
                "{join:?} {variant:?}"
            );
        }
    }
    assert_eq!(top_positions(&remote, 2), top_positions(&local, 2));
}

/// Recurring calls inside a larger graph are fused into an op-graph the client caches, so the
/// operation reaches the server as the IR the Fusion wrapper built rather than through the
/// router client: both must carry the whole call.
#[test]
fn generated_remote_extension_runs_inside_a_cached_graph() {
    let remote = remote_device();
    let local = Device::flex();
    let run = |device: &Device| {
        let mut x = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], device);
        for step in 0..8 {
            let operand = Operand::Scaled {
                values: (x.clone() * 0.5).into_dispatch(),
                factor: Factor(step as f32),
            };
            let summary = <Dispatch as ExampleBackend>::combine(
                (x.clone() + 1.0).into_dispatch(),
                -0.5,
                Join::Mul,
                operand,
            );
            let positive = Tensor::<2, Int>::from_dispatch(summary.positive).float();
            x = (Tensor::<2>::from_dispatch(summary.combined) * 0.1 + positive).tanh();
        }
        x.into_data().try_to_vec::<f32>().unwrap()
    };
    let (remote, local) = (run(&remote), run(&local));
    for (r, l) in remote.iter().zip(&local) {
        assert!((r - l).abs() < 1e-5, "remote {remote:?} local {local:?}");
    }
}

/// The default operation ids name the trait, so two traits' `forward` handlers share a registry
/// without either replacing the other; registering a trait twice is harmless.
#[test]
fn same_named_methods_of_two_traits_share_a_registry() {
    let mut registry = CustomOpRegistry::<Flex>::new();
    ShiftBackendHandlers::register(&mut registry);
    RepeatBackendHandlers::register(&mut registry);
    ShiftBackendHandlers::register(&mut registry);
    let remote = serve(registry);
    let local = Device::flex();
    let run = |device: &Device| {
        let x = Tensor::<1>::from_floats([1.0, -2.0, 3.0], device).into_dispatch();
        let shifted = <Dispatch as ShiftBackend>::forward(x, 0.5, true);
        let repeated = <Dispatch as RepeatBackend>::forward(shifted, 3);
        Tensor::<1>::from_dispatch(repeated)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    };
    assert_eq!(run(&local), vec![-4.5, 4.5, -10.5]);
    assert_eq!(run(&remote), run(&local));
}

#[test]
#[should_panic(expected = "already registered for custom operation `ShiftBackend::forward`")]
fn a_different_handler_under_a_taken_id_is_refused() {
    let mut registry = CustomOpRegistry::<Flex>::new();
    ShiftBackendHandlers::register(&mut registry);
    ClashingBackendHandlers::register(&mut registry);
}
