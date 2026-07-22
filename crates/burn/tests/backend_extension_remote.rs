//! Verifies that `#[backend_extension(Remote)]` expands into a valid `impl ... for Dispatch`,
//! including for operations with no tensor inputs (where the backend can't be selected from an
//! input tensor — the single declared backend is used directly).
//!
//! The client-side `impl ... for Remote` is hand-written by the user; here it is stubbed, since the
//! test only checks that the generated dispatch glue type-checks.
#![cfg(feature = "remote")]

use burn::backend::{
    Dispatch, ExtensionType, Remote, backend_extension,
    tensor::{FloatTensor, IntTensor},
};
// Autodiff is enabled transitively by the default features in this test build. Several traits below
// list `Autodiff`, whose generated arms reference the bare `Autodiff` type.
#[cfg(feature = "autodiff")]
use burn::backend::Autodiff;

#[derive(ExtensionType)]
pub struct Pair<B: burn::backend::Backend> {
    pub a: FloatTensor<B>,
    pub b: IntTensor<B>,
}

// Tuple struct: exercises the derive's unnamed-field (positional) handling.
#[derive(ExtensionType)]
pub struct TupleInputs<B: burn::backend::Backend>(pub FloatTensor<B>, pub IntTensor<B>);

// Enum input: a tensor-tuple variant, a named-fields variant, and a tensor-less unit variant. The
// unit variant has no representative tensor, so `dispatch_repr`/`dispatch_float_repr` return `None`
// for it and backend selection must defer to another input (or panic if there is none).
#[derive(ExtensionType)]
pub enum Operand<B: burn::backend::Backend> {
    Dense(FloatTensor<B>),
    Sparse {
        values: FloatTensor<B>,
        indices: IntTensor<B>,
    },
    Empty,
}

// Nested `#[extension_type]` field plus a non-tensor passthrough field, mirroring burn-vision's
// `ConnectedStatsPrimitive`. Guards the derive's recursion and passthrough handling for the new
// enum-capable codegen (used as both output and input below).
#[derive(ExtensionType)]
pub struct Coords<B: burn::backend::Backend> {
    pub left: IntTensor<B>,
    pub top: IntTensor<B>,
}

#[derive(ExtensionType)]
pub struct Stats<B: burn::backend::Backend> {
    pub area: IntTensor<B>,
    #[extension_type]
    pub coords: Coords<B>,
    pub count: usize,
}

#[backend_extension(Remote)]
pub trait Backend: burn::backend::Backend {
    fn load_data(data_index: usize) -> FloatTensor<Self>;
    fn scale(x: FloatTensor<Self>, factor: f32) -> FloatTensor<Self>;
    fn split(x: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>);
    fn make_pair(x: FloatTensor<Self>) -> Pair<Self>;
    // Struct-of-tensors as an *input*: the `#[extension_type]` marker tells the macro to unwrap the
    // incoming `Pair<Dispatch>` back into `Pair<Remote>` before the backend call.
    fn combine_pair(#[extension_type] pair: Pair<Self>, factor: f32) -> FloatTensor<Self>;
    // A struct input mixed with a bare tensor input: the backend is selected from the bare tensor,
    // and the struct is unwrapped for that same backend inside the arm.
    fn mix(x: FloatTensor<Self>, #[extension_type] pair: Pair<Self>) -> FloatTensor<Self>;
    // Multiple struct inputs: each is unwrapped independently for the selected backend.
    fn combine_two(
        #[extension_type] lhs: Pair<Self>,
        #[extension_type] rhs: Pair<Self>,
    ) -> FloatTensor<Self>;
}

// User-written client side: builds a `CustomOpIr` and ships it to the server. Stubbed here.
impl Backend for Remote {
    fn load_data(_data_index: usize) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
    fn scale(_x: FloatTensor<Self>, _factor: f32) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
    fn split(_x: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>) {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
    fn make_pair(_x: FloatTensor<Self>) -> Pair<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
    fn combine_pair(_pair: Pair<Self>, _factor: f32) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
    fn mix(_x: FloatTensor<Self>, _pair: Pair<Self>) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
    fn combine_two(_lhs: Pair<Self>, _rhs: Pair<Self>) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
}

#[test]
fn remote_backend_extension_dispatch_compiles() {
    // Referencing the generated dispatch methods proves the macro expanded into a well-typed
    // `impl Backend for Dispatch` for both the no-tensor-input `load_data` op and the
    // tensor-input `scale` op.
    let _f: fn(usize) -> FloatTensor<Dispatch> = <Dispatch as Backend>::load_data;
    let _g: fn(FloatTensor<Dispatch>, f32) -> FloatTensor<Dispatch> = <Dispatch as Backend>::scale;
    // The struct-input op expands into a well-typed dispatch method: it takes the dispatch form
    // `Pair<Dispatch>`, peeks its backend tag, and unwraps to `Pair<Remote>` before the call.
    let _h: fn(Pair<Dispatch>, f32) -> FloatTensor<Dispatch> = <Dispatch as Backend>::combine_pair;
    // Mixed bare-tensor + struct input, and multiple struct inputs, both expand to well-typed glue.
    let _i: fn(FloatTensor<Dispatch>, Pair<Dispatch>) -> FloatTensor<Dispatch> =
        <Dispatch as Backend>::mix;
    let _j: fn(Pair<Dispatch>, Pair<Dispatch>) -> FloatTensor<Dispatch> =
        <Dispatch as Backend>::combine_two;
    // Struct input on an op whose `Autodiff` entry is `cfg`-gated off in this build. The generated
    // autodiff arms must carry that cfg and be stripped; otherwise they reference a bare `Autodiff`
    // that isn't in scope here. This compiling proves the mixed path honors the autodiff cfg gate
    // (regression test for the review's finding #1). (See `AdGatedBackend`.)
    let _k: fn(Pair<Dispatch>) -> FloatTensor<Dispatch> = <Dispatch as AdGatedBackend>::ad_gated;
}

// Struct input combined with a `cfg`-gated `Autodiff`. The gate `cfg(not(feature = "remote"))` is
// always false in this `#![cfg(feature = "remote")]` build, so every generated autodiff arm must be
// stripped. If the mixed path failed to propagate the autodiff cfg (the pre-fix bug), the arms would
// remain and reference the unimported `Autodiff` type, failing to compile — so this trait building at
// all is the assertion. Mirrors `GatedBackend`, but exercises the autodiff-gate path specifically.
#[backend_extension(Autodiff: cfg(not(feature = "remote")), Remote)]
pub trait AdGatedBackend: burn::backend::Backend {
    fn ad_gated(#[extension_type] pair: Pair<Self>) -> FloatTensor<Self>;
}

impl AdGatedBackend for Remote {
    fn ad_gated(_pair: Pair<Self>) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
}

// Only referenced by the generated autodiff arms, which are gated off in this build.
#[cfg(not(feature = "remote"))]
impl AdGatedBackend for burn::backend::Autodiff<Remote> {
    fn ad_gated(_pair: Pair<Self>) -> FloatTensor<Self> {
        unimplemented!("would register the backward pass")
    }
}

// Nested `#[extension_type]` struct as both an output and an input, with a passthrough field.
#[backend_extension(Remote)]
pub trait NestedBackend: burn::backend::Backend {
    fn make_stats(x: FloatTensor<Self>) -> Stats<Self>;
    fn use_stats(#[extension_type] s: Stats<Self>) -> IntTensor<Self>;
}

impl NestedBackend for Remote {
    fn make_stats(_x: FloatTensor<Self>) -> Stats<Self> {
        unimplemented!("stub")
    }
    fn use_stats(_s: Stats<Self>) -> IntTensor<Self> {
        unimplemented!("stub")
    }
}

#[test]
fn nested_extension_type_dispatch_compiles() {
    let _a: fn(FloatTensor<Dispatch>) -> Stats<Dispatch> = <Dispatch as NestedBackend>::make_stats;
    let _b: fn(Stats<Dispatch>) -> IntTensor<Dispatch> = <Dispatch as NestedBackend>::use_stats;
}

// Enum and tuple-struct inputs, including mixing an enum with a bare tensor. Combined with
// `Autodiff` to exercise the autodiff enum unwrap (per-variant, per-field float nesting).
#[backend_extension(Autodiff, Remote)]
pub trait EnumBackend: burn::backend::Backend {
    fn use_operand(#[extension_type] op: Operand<Self>) -> FloatTensor<Self>;
    fn use_tuple(#[extension_type] inp: TupleInputs<Self>) -> FloatTensor<Self>;
    // Enum mixed with a bare tensor: if the enum lands on `Empty`, the backend is selected from `x`.
    fn mix_enum(x: FloatTensor<Self>, #[extension_type] op: Operand<Self>) -> FloatTensor<Self>;
}

impl EnumBackend for Remote {
    fn use_operand(_op: Operand<Self>) -> FloatTensor<Self> {
        unimplemented!("stub")
    }
    fn use_tuple(_inp: TupleInputs<Self>) -> FloatTensor<Self> {
        unimplemented!("stub")
    }
    fn mix_enum(_x: FloatTensor<Self>, _op: Operand<Self>) -> FloatTensor<Self> {
        unimplemented!("stub")
    }
}

#[cfg(feature = "autodiff")]
impl EnumBackend for burn::backend::Autodiff<Remote> {
    fn use_operand(_op: Operand<Self>) -> FloatTensor<Self> {
        unimplemented!("would register the backward pass over the active variant's tracked fields")
    }
    fn use_tuple(_inp: TupleInputs<Self>) -> FloatTensor<Self> {
        unimplemented!("stub")
    }
    fn mix_enum(_x: FloatTensor<Self>, _op: Operand<Self>) -> FloatTensor<Self> {
        unimplemented!("stub")
    }
}

#[test]
fn enum_and_tuple_input_dispatch_compiles() {
    // Enum input, tuple-struct input, and enum-mixed-with-bare-tensor input all expand into
    // well-typed dispatch methods routing concrete `Remote` and autodiff `Autodiff<Remote>`.
    let _a: fn(Operand<Dispatch>) -> FloatTensor<Dispatch> = <Dispatch as EnumBackend>::use_operand;
    let _b: fn(TupleInputs<Dispatch>) -> FloatTensor<Dispatch> =
        <Dispatch as EnumBackend>::use_tuple;
    let _c: fn(FloatTensor<Dispatch>, Operand<Dispatch>) -> FloatTensor<Dispatch> =
        <Dispatch as EnumBackend>::mix_enum;
}

// A no-tensor-input op on a `cfg`-gated single backend. The backend is gated on
// `cfg(not(feature = "remote"))`, which is always false in this `#![cfg(feature = "remote")]` test
// build, so the backend is compiled out — exercising the macro's no-input + gated codegen path,
// where the call must be gated on the backend's cfg with an `unimplemented!` fallback (rather than
// referencing a backend that doesn't exist).
#[backend_extension(Remote: cfg(not(feature = "remote")))]
pub trait GatedBackend: burn::backend::Backend {
    // Underscore-prefixed because, in this test, the backend is always compiled out, so the
    // generated dispatch body never uses the argument.
    fn gated_load(_data_index: usize) -> FloatTensor<Self>;
}

impl GatedBackend for Remote {
    fn gated_load(_data_index: usize) -> FloatTensor<Self> {
        unimplemented!("the client builds a CustomOpIr and ships it to the server")
    }
}

#[test]
#[should_panic(expected = "Backend not supported for custom op `gated_load`")]
fn remote_backend_extension_gated_no_input_falls_back() {
    // With the only backend compiled out, the generated dispatch body takes the `unimplemented!`
    // fallback arm. Reaching it (rather than a compile error) proves the gated no-input path
    // expanded into valid code.
    let _ = <Dispatch as GatedBackend>::gated_load(0);
}

// Struct inputs combined with `Autodiff`. The generated dispatch glue must, for the autodiff arm,
// peel the `Autodiff(..)` nesting on each float field to rebuild `Pair<Autodiff<Remote>>`, call the
// user's `impl ... for Autodiff<Remote>`, and re-wrap the output as an autodiff dispatch tensor.
#[cfg(feature = "autodiff")]
mod autodiff_struct_input {
    use super::*;

    #[backend_extension(Autodiff, Remote)]
    pub trait AdBackend: burn::backend::Backend {
        fn ad_combine(#[extension_type] pair: Pair<Self>, factor: f32) -> FloatTensor<Self>;
        // Struct mixed with a bare (autodiff-tracked) float tensor.
        fn ad_mix(x: FloatTensor<Self>, #[extension_type] pair: Pair<Self>) -> FloatTensor<Self>;
    }

    // Client side for the plain remote backend (stubbed).
    impl AdBackend for Remote {
        fn ad_combine(_pair: Pair<Self>, _factor: f32) -> FloatTensor<Self> {
            unimplemented!("the client builds a CustomOpIr and ships it to the server")
        }
        fn ad_mix(_x: FloatTensor<Self>, _pair: Pair<Self>) -> FloatTensor<Self> {
            unimplemented!("stub")
        }
    }

    // User-written autodiff side: normally registers a `Backward` step; stubbed here since the test
    // only checks that the generated dispatch glue type-checks.
    impl AdBackend for Autodiff<Remote> {
        fn ad_combine(_pair: Pair<Self>, _factor: f32) -> FloatTensor<Self> {
            unimplemented!("would register the backward pass over the struct's tracked fields")
        }
        fn ad_mix(_x: FloatTensor<Self>, _pair: Pair<Self>) -> FloatTensor<Self> {
            unimplemented!("stub")
        }
    }

    #[test]
    fn autodiff_struct_input_dispatch_compiles() {
        // Referencing the dispatch methods proves the macro expanded a well-typed `impl for Dispatch`
        // whose match routes both concrete (`Remote`) and autodiff (`Autodiff<Remote>`) inputs.
        let _a: fn(Pair<Dispatch>, f32) -> FloatTensor<Dispatch> =
            <Dispatch as AdBackend>::ad_combine;
        let _b: fn(FloatTensor<Dispatch>, Pair<Dispatch>) -> FloatTensor<Dispatch> =
            <Dispatch as AdBackend>::ad_mix;
    }
}
