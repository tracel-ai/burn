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

#[derive(ExtensionType)]
pub struct Pair<B: burn::backend::Backend> {
    pub a: FloatTensor<B>,
    pub b: IntTensor<B>,
}

#[backend_extension(Remote)]
pub trait Backend: burn::backend::Backend {
    fn load_data(data_index: usize) -> FloatTensor<Self>;
    fn scale(x: FloatTensor<Self>, factor: f32) -> FloatTensor<Self>;
    fn split(x: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>);
    fn make_pair(x: FloatTensor<Self>) -> Pair<Self>;
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
}

#[test]
fn remote_backend_extension_dispatch_compiles() {
    // Referencing the generated dispatch methods proves the macro expanded into a well-typed
    // `impl Backend for Dispatch` for both the no-tensor-input `load_data` op and the
    // tensor-input `scale` op.
    let _f: fn(usize) -> FloatTensor<Dispatch> = <Dispatch as Backend>::load_data;
    let _g: fn(FloatTensor<Dispatch>, f32) -> FloatTensor<Dispatch> = <Dispatch as Backend>::scale;
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
