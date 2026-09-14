#![cfg(all(feature = "extension", feature = "cpu", feature = "fusion"))]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Backend, Dispatch, ExtensionType, backend_extension,
    ops::{FloatTensorOps, IntTensorOps},
    tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
};
use burn::tensor::{Bool, Device, Int, Tensor};
use burn_cubecl::CubeBackend;

mod inner {
    use super::*;

    #[derive(ExtensionType)]
    #[extension_type(fusion)]
    pub struct Inner<B: Backend> {
        pub integer: IntTensor<B>,
        pub mask: BoolTensor<B>,
    }
}
use inner::Inner;
#[derive(ExtensionType)]
#[extension_type(fusion)]
pub struct Outputs<B: Backend> {
    pub float: FloatTensor<B>,
    #[extension_type]
    pub inner: Inner<B>,
}

// Both imports must work without importing or renaming the companion metadata type.
mod renamed {
    use super::*;
    use inner::Inner as RenamedInner;

    #[derive(ExtensionType)]
    #[extension_type(fusion)]
    pub struct Outputs<B: Backend> {
        #[extension_type]
        pub inner: RenamedInner<B>,
    }
}

#[test]
fn nested_metadata_resolves_renamed_imports() {
    use burn::backend::fusion::custom::FusionValueAdapter;

    let spec = TensorSpec::new([3].into(), burn::tensor::DType::I32);
    let metadata = renamed::OutputsMetadata {
        inner: inner::InnerMetadata {
            integer: spec.clone(),
            mask: TensorSpec::new(
                [3].into(),
                burn::tensor::DType::Bool(burn::tensor::BoolStore::Native),
            ),
        },
    };
    let mut specs = Vec::new();
    <renamed::Outputs<CubeBackend> as FusionValueAdapter<CubeBackend>>::append_output_specs(
        &metadata, &mut specs,
    );
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].shape, spec.shape);
    assert_eq!(specs[0].dtype, spec.dtype);
    assert!(specs[1].dtype.is_bool());
}
fn metadata(
    x: &TensorSpec,
    _alias: &TensorSpec,
    integer: &TensorSpec,
    mask: &TensorSpec,
    _amount: &f32,
    wrong: &bool,
) -> OutputsMetadata {
    let mut float = x.clone();
    if *wrong {
        float.shape = [999].into();
    }
    OutputsMetadata {
        float,
        inner: inner::InnerMetadata {
            integer: integer.clone(),
            mask: mask.clone(),
        },
    }
}
type Amount = f32;

#[backend_extension(Cube, Fusion)]
pub trait TestOps: Backend {
    #[cfg(any())]
    fn disabled_without_annotation(x: FloatTensor<Self>) -> FloatTensor<Self>;

    #[fusion(meta = |x| (x.clone(), x.clone()))]
    fn aliases(x: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(dtype = x, shape = x)]
    fn quantized(x: QuantizedTensor<Self>) -> QuantizedTensor<Self>;

    #[fusion(dtype = x, shape = x)]
    fn add(x: &FloatTensor<Self>, #[fusion(scalar)] amount: Amount) -> FloatTensor<Self>;

    #[fusion(shape = joined_shape(lhs, rhs, axis), dtype = lhs)]
    fn join(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, axis: usize) -> FloatTensor<Self>;

    #[fusion(meta = metadata)]
    fn mixed(
        x: &FloatTensor<Self>,
        alias: FloatTensor<Self>,
        integer: IntTensor<Self>,
        mask: BoolTensor<Self>,
        amount: f32,
        wrong: bool,
    ) -> Outputs<Self>;

    #[fusion(default)]
    fn inherited(x: FloatTensor<Self>) -> FloatTensor<Self> {
        x
    }
}

fn joined_shape(
    lhs: &burn::tensor::Shape,
    rhs: &burn::tensor::Shape,
    axis: &usize,
) -> burn::tensor::Shape {
    let mut shape = lhs.clone();
    shape[*axis] += rhs[*axis];
    shape
}

impl TestOps for CubeBackend {
    fn add(x: &FloatTensor<Self>, amount: f32) -> FloatTensor<Self> {
        Self::float_add_scalar(x.clone(), amount.into())
    }

    fn join(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, axis: usize) -> FloatTensor<Self> {
        Self::float_cat(vec![lhs, rhs], axis)
    }

    fn aliases(x: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>) {
        (x.clone(), x)
    }
    fn quantized(x: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        x
    }

    fn mixed(
        x: &FloatTensor<Self>,
        alias: FloatTensor<Self>,
        integer: IntTensor<Self>,
        mask: BoolTensor<Self>,
        amount: f32,
        _wrong: bool,
    ) -> Outputs<Self> {
        drop(alias);
        Outputs {
            float: Self::float_add_scalar(x.clone(), amount.into()),
            inner: Inner { integer, mask },
        }
    }
}

#[test]
fn field_expressions_copy_and_compute_shapes() {
    let device = Device::cpu();
    let lhs = Tensor::<1>::from_floats([1., 2.], &device).into_dispatch();
    let rhs = Tensor::<1>::from_floats([3.], &device).into_dispatch();
    let added = Dispatch::add(&lhs, 2.);
    let joined = Tensor::<1>::from_dispatch(Dispatch::join(added, rhs, 0));
    assert_eq!(joined.dims(), [3]);
    joined
        .into_data()
        .assert_eq(&burn::tensor::TensorData::from([3., 4., 3.]), false);
    Tensor::<1>::from_dispatch(lhs)
        .into_data()
        .assert_eq(&burn::tensor::TensorData::from([1., 2.]), false);
}
#[test]
fn nested_mixed_borrowed_aliases_streams_and_options() {
    let device = Device::cpu();
    let x = Tensor::<1>::from_floats([1., 2., 3.], &device).into_dispatch();
    let integer = std::thread::spawn(|| {
        Tensor::<1, Int>::from_ints([4, 5, 6], &Device::cpu()).into_dispatch()
    })
    .join()
    .unwrap();
    let mask = Tensor::<1, Bool>::from_bool([true, false, true], &device).into_dispatch();
    for amount in [2., 7.] {
        let out = Dispatch::mixed(&x, x.clone(), integer.clone(), mask.clone(), amount, false);
        let float = Dispatch::inherited(out.float);
        let float = Tensor::<1>::from_dispatch(float) + 1.;
        float.into_data().assert_eq(
            &burn::tensor::TensorData::from([amount + 2., amount + 3., amount + 4.]),
            false,
        );
        Tensor::<1, Int>::from_dispatch(out.inner.integer)
            .into_data()
            .assert_eq(&burn::tensor::TensorData::from([4, 5, 6]), false);
        Tensor::<1, Bool>::from_dispatch(out.inner.mask)
            .into_data()
            .assert_eq(&burn::tensor::TensorData::from([true, false, true]), false);
    }
}

#[test]
fn incorrect_metadata_is_an_execution_error() {
    let device = Device::cpu();
    let x = Tensor::<1>::from_floats([1.], &device).into_dispatch();
    let out = Dispatch::mixed(
        &x,
        x.clone(),
        Tensor::<1, Int>::from_ints([1], &device).into_dispatch(),
        Tensor::<1, Bool>::from_bool([true], &device).into_dispatch(),
        0.,
        true,
    );
    let error = burn::tensor::read_sync(Dispatch::float_into_data(out.float)).unwrap_err();
    assert!(error.to_string().contains("metadata mismatch"), "{error}");
    let error = burn::tensor::read_sync(Dispatch::int_into_data(out.inner.integer)).unwrap_err();
    assert!(error.to_string().contains("metadata mismatch"), "{error}");
}

#[test]
fn aliased_outputs_can_feed_independent_consumers() {
    let x = Tensor::<1>::from_floats([1., 2.], &Device::cpu());
    let (a, b) = Dispatch::aliases(x.into_dispatch());
    let a = Tensor::<1>::from_dispatch(a) + 1.;
    let b = Tensor::<1>::from_dispatch(b) * 2.;
    (a + b)
        .into_data()
        .assert_eq(&burn::tensor::TensorData::from([4., 7.]), false);
}

#[test]
fn rejects_incorrect_dtype_category_before_registration() {
    use burn::backend::fusion::custom::{Float, FusionValueAdapter};
    let mut specs = Vec::new();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <Float as FusionValueAdapter<CubeBackend>>::append_output_specs(
                &TensorSpec::new([1].into(), burn::tensor::DType::I32),
                &mut specs,
            );
        }))
        .is_err()
    );
    assert!(specs.is_empty());
}

#[derive(ExtensionType)]
#[extension_type(fusion)]
pub enum Packet<B: Backend> {
    Empty,
    Float { tensor: FloatTensor<B>, mode: u32 },
    Nested(#[extension_type] Outputs<B>, u32),
}

#[backend_extension(Cube, Fusion)]
pub trait StructuredOps: Backend {
    #[fusion(meta = |first, second, _wrong| (first.clone(), second.clone()))]
    fn relay(
        #[extension_type] first: Packet<Self>,
        #[extension_type] second: Packet<Self>,
        wrong: u32,
    ) -> (Packet<Self>, Packet<Self>);
}
impl StructuredOps for CubeBackend {
    fn relay(
        first: Packet<Self>,
        mut second: Packet<Self>,
        wrong: u32,
    ) -> (Packet<Self>, Packet<Self>) {
        if wrong == 1 {
            return (Packet::Empty, second);
        }
        if wrong == 2 {
            return (first, Packet::Empty);
        }
        if let Packet::Nested(_, mode) = &mut second {
            *mode += 1;
        }
        (first, second)
    }
}

#[test]
fn nested_enum_inputs_round_trip_with_empty_variants_and_scalar_fields() {
    let device = Device::cpu();
    let packet = Packet::Nested(
        Outputs {
            float: Tensor::<1>::from_floats([1., 2.], &device).into_dispatch(),
            inner: Inner {
                integer: Tensor::<1, Int>::from_ints([3, 4], &device).into_dispatch(),
                mask: Tensor::<1, Bool>::from_bool([true, false], &device).into_dispatch(),
            },
        },
        7,
    );
    let (empty, result) = Dispatch::relay(Packet::Empty, packet, 0);
    assert!(matches!(empty, Packet::Empty));
    let Packet::Nested(out, mode) = result else {
        panic!("wrong variant")
    };
    // Execution changes this field, but the lazy result takes its value from metadata.
    assert_eq!(mode, 7);
    Tensor::<1>::from_dispatch(out.float)
        .into_data()
        .assert_eq(&burn::tensor::TensorData::from([1., 2.]), false);
    Tensor::<1, Int>::from_dispatch(out.inner.integer)
        .into_data()
        .assert_eq(&burn::tensor::TensorData::from([3, 4]), false);
    Tensor::<1, Bool>::from_dispatch(out.inner.mask)
        .into_data()
        .assert_eq(&burn::tensor::TensorData::from([true, false]), false);
}

#[test]
fn enum_variant_mismatches_fail_before_any_output_is_published() {
    // Independent threads isolate streams after each intentional execution error.
    for wrong in [1, 2] {
        std::thread::spawn(move || {
            let device = Device::cpu();
            let packet = || Packet::Float {
                tensor: Tensor::<1>::from_floats([1.], &device).into_dispatch(),
                mode: 3,
            };
            let (first, second) = Dispatch::relay(packet(), packet(), wrong);
            for value in [first, second] {
                let Packet::Float { tensor, .. } = value else {
                    panic!("metadata determines the lazy variant")
                };
                let error = burn::tensor::read_sync(Dispatch::float_into_data(tensor)).unwrap_err();
                assert!(
                    error.to_string().contains("differs from its metadata"),
                    "{error}"
                );
            }
        })
        .join()
        .unwrap();
    }
}
