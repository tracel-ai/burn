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
    fusion::custom::{ExtensionMetadata, TensorSpec},
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

/// A tuple struct with a nested extension value and a positional, custom-encoded field.
#[derive(ExtensionType)]
#[extension_type(fusion, remote)]
pub struct Weighted<B: Backend>(
    #[extension_type] pub Operand<B>,
    #[extension_type(scalar = value.0, decode = Factor)] pub Factor,
);

/// An enum output whose variant the metadata selects from an ordinary argument.
#[derive(ExtensionType)]
#[extension_type(fusion, remote)]
pub enum Picked<B: Backend> {
    One(FloatTensor<B>),
    Two {
        values: FloatTensor<B>,
        positive: IntTensor<B>,
    },
}

/// Generated behind a `Remote` condition, which holds in this test.
#[backend_extension(Flex, Fusion, Remote: cfg(feature = "remote-websocket"))]
pub trait LayoutBackend: Backend {
    /// `x` plus the weighted operand: `One` alone, or `Two` with which elements are positive.
    #[fusion(meta = |x, _weighted, two| if *two {
        PickedMetadata::Two {
            values: x.clone(),
            positive: TensorSpec::new(x.shape.clone(), DType::I64),
        }
    } else {
        PickedMetadata::One(x.clone())
    })]
    fn pick(
        x: &FloatTensor<Self>,
        #[extension_type] weighted: Weighted<Self>,
        two: bool,
    ) -> Picked<Self>;

    /// `x` negated, and which elements of `x` are positive; a tuple output.
    #[fusion(meta = |x| (x.clone(), TensorSpec::new(x.shape.clone(), DType::I64)))]
    fn split(x: &FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>);

    /// `x` plus the operand's values; the metadata names `Self`.
    #[fusion(meta = |x: &TensorSpec, _operand: &<Operand<Self> as ExtensionMetadata>::Metadata| x.clone())]
    fn offset(x: FloatTensor<Self>, #[extension_type] operand: Operand<Self>) -> FloatTensor<Self>;
}

/// An enum output whose variants hold the same tensors: only the variant tells them apart.
#[derive(ExtensionType)]
#[extension_type(fusion, remote)]
pub enum Signed<B: Backend> {
    Kept(FloatTensor<B>),
    Negated(FloatTensor<B>),
}

/// Describes outputs other than the ones the backend returns.
#[backend_extension(Flex, Fusion, Remote)]
pub trait MisdescribedBackend: Backend {
    /// Described as `Kept` while the backend returns `Negated`.
    #[fusion(meta = |x| SignedMetadata::Kept(x.clone()))]
    fn misdescribed(x: &FloatTensor<Self>) -> Signed<Self>;

    /// Described as `I64` while the backend returns `x` itself, a float tensor.
    #[fusion(meta = |x| TensorSpec::new(x.shape.clone(), DType::I64))]
    fn misdescribed_dtype(x: &FloatTensor<Self>) -> IntTensor<Self>;
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
        let other = operand_values(operand);
        let joined = match join {
            Join::Add => Flex::float_add(x, other),
            Join::Mul => Flex::float_mul(x, other),
        };
        let combined = Flex::float_add_scalar(joined, bias.into());
        let positive = positive(combined.clone());
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

impl LayoutBackend for Flex {
    fn pick(x: &FloatTensor<Self>, weighted: Weighted<Self>, two: bool) -> Picked<Self> {
        let Weighted(operand, factor) = weighted;
        let weight = Flex::float_mul_scalar(operand_values(operand), factor.0.into());
        let values = Flex::float_add(x.clone(), weight);
        if two {
            Picked::Two {
                positive: positive(values.clone()),
                values,
            }
        } else {
            Picked::One(values)
        }
    }

    fn split(x: &FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>) {
        (Flex::float_neg(x.clone()), positive(x.clone()))
    }

    fn offset(x: FloatTensor<Self>, operand: Operand<Self>) -> FloatTensor<Self> {
        Flex::float_add(x, operand_values(operand))
    }
}

impl MisdescribedBackend for Flex {
    fn misdescribed(x: &FloatTensor<Self>) -> Signed<Self> {
        Signed::Negated(Flex::float_neg(x.clone()))
    }

    fn misdescribed_dtype(x: &FloatTensor<Self>) -> IntTensor<Self> {
        x.clone()
    }
}

/// The operand's values, with the mask and fill or the factor applied.
fn operand_values(operand: Operand<Flex>) -> FloatTensor<Flex> {
    match operand {
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
    }
}

/// One where an element is positive, zero elsewhere.
fn positive(values: FloatTensor<Flex>) -> IntTensor<Flex> {
    Flex::bool_into_int(
        Flex::float_greater_elem(values, 0.0.into(), BoolDType::Native),
        IntDType::I64,
    )
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
    LayoutBackendHandlers::register(&mut registry);
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

/// A remote device whose server serves [`MisdescribedBackend`].
fn misdescribed_device() -> Device {
    let mut registry = CustomOpRegistry::<Flex>::new();
    MisdescribedBackendHandlers::register(&mut registry);
    serve(registry)
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
    let operand = operand(device, variant);
    let summary = <Dispatch as ExampleBackend>::combine(x.into_dispatch(), 0.25, join, operand);
    Combined {
        combined: floats(summary.combined),
        positive: ints(summary.positive),
        count: summary.count,
    }
}

fn operand(device: &Device, variant: Variant) -> Operand<Dispatch> {
    let values = Tensor::<2>::from_floats([[0.5, 1.5, -2.5], [3.5, -4.5, 5.5]], device);
    match variant {
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
    }
}

/// What `pick` returns, read back to the host.
#[derive(Debug, PartialEq)]
enum PickedValues {
    One(Vec<f32>),
    Two(Vec<f32>, Vec<i64>),
}

fn pick(device: &Device, variant: Variant, two: bool) -> PickedValues {
    let x = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], device);
    let weighted = Weighted(operand(device, variant), Factor(0.5));
    match <Dispatch as LayoutBackend>::pick(&x.into_dispatch(), weighted, two) {
        Picked::One(values) => PickedValues::One(floats(values)),
        Picked::Two { values, positive } => PickedValues::Two(floats(values), ints(positive)),
    }
}

fn split(device: &Device) -> (Vec<f32>, Vec<i64>) {
    let x = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], device);
    let (negated, positive) = <Dispatch as LayoutBackend>::split(&x.into_dispatch());
    (floats(negated), ints(positive))
}

fn offset(device: &Device) -> Vec<f32> {
    let x = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], device);
    let operand = operand(device, Variant::Masked);
    floats(<Dispatch as LayoutBackend>::offset(
        x.into_dispatch(),
        operand,
    ))
}

fn floats(tensor: FloatTensor<Dispatch>) -> Vec<f32> {
    Tensor::<2>::from_dispatch(tensor)
        .into_data()
        .try_to_vec()
        .unwrap()
}

fn ints(tensor: IntTensor<Dispatch>) -> Vec<i64> {
    Tensor::<2, Int>::from_dispatch(tensor)
        .into_data()
        .try_to_vec()
        .unwrap()
}

fn top_positions(device: &Device, k: usize) -> Vec<i64> {
    let scores = Tensor::<2>::from_floats([[0.1, 0.9, 0.5, 0.7], [0.8, 0.2, 0.6, 0.4]], device);
    let positions = <Dispatch as ExampleBackend>::top_positions(scores.into_dispatch(), k);
    Tensor::<2, Int>::from_dispatch(positions)
        .into_data()
        .try_to_vec()
        .unwrap()
}

/// The server rebuilds every argument from the IR alone: each ordinary argument, extension
/// variant and field, scalar or tensor, reaches the backend with the value the client passed, so
/// the remote result equals the local one.
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

/// Every output shape travels: both variants of an enum output, as the metadata selects them, and
/// a tuple, with a borrowed input and a tuple struct carrying a nested extension value and a
/// positional scalar field, all from a trait whose `Remote` is behind a condition.
#[test]
fn generated_remote_extension_returns_structured_outputs() {
    let remote = remote_device();
    let local = Device::flex();
    for variant in [Variant::Dense, Variant::Masked, Variant::Scaled] {
        for two in [false, true] {
            let picked = pick(&remote, variant, two);
            assert_eq!(matches!(picked, PickedValues::Two(..)), two, "{picked:?}");
            assert_eq!(picked, pick(&local, variant, two), "{variant:?} {two}");
        }
    }
    assert_eq!(split(&remote), split(&local));
}

/// The server never evaluates the metadata callable, so one whose types name `Self` builds and
/// runs with `Remote` listed: the handlers' impl block has a `Self` of its own, and metadata run
/// there would not compile.
#[test]
fn metadata_naming_self_runs_remotely() {
    assert_eq!(offset(&remote_device()), offset(&Device::flex()));
}

/// The server checks the backend's output against the IR in every build: an enum in another
/// variant than the client recorded is refused, even when both variants hold the same tensors, so
/// reading the output fails rather than returning a value the client takes for the other variant.
#[test]
fn an_output_in_another_variant_than_the_ir_records_is_refused() {
    let x = Tensor::<2>::from_floats([[1.0, -2.0], [3.0, -4.0]], &misdescribed_device());
    let Signed::Kept(values) = <Dispatch as MisdescribedBackend>::misdescribed(&x.into_dispatch())
    else {
        unreachable!("the client builds the variant its metadata describes");
    };
    let error = Tensor::<2>::from_dispatch(values)
        .try_into_data()
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(
            "custom operation `MisdescribedBackend::misdescribed`: the backend returned a `Signed` \
             in variant 1 where the IR records variant 0"
        ),
        "{error}"
    );
}

/// A tensor in another dtype than its output IR describes is refused in release builds too, so
/// reading it fails rather than reinterpreting its bytes.
#[test]
fn an_output_in_another_dtype_than_the_ir_describes_is_refused() {
    let x = Tensor::<2>::from_floats([[1.0, -2.0], [3.0, -4.0]], &misdescribed_device());
    let values = <Dispatch as MisdescribedBackend>::misdescribed_dtype(&x.into_dispatch());
    let error = Tensor::<2, Int>::from_dispatch(values)
        .try_into_data()
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("the backend returned a F32 output where the IR describes I64"),
        "{error}"
    );
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

/// A registry refuses a second handler under an id another trait's method already claims, rather
/// than serving one trait's calls with the other's handler.
#[test]
#[should_panic(expected = "already registered for custom operation `ShiftBackend::forward`")]
fn a_different_handler_under_a_taken_id_is_refused() {
    let mut registry = CustomOpRegistry::<Flex>::new();
    ShiftBackendHandlers::register(&mut registry);
    ClashingBackendHandlers::register(&mut registry);
}
