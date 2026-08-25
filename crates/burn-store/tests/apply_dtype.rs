//! The applier checks that a loaded tensor's dtype is of the right *kind* for the parameter
//! it is going into, not only that the shapes agree. Widths are deliberately left alone, so a
//! half-precision or int64 source still loads as itself. Regression tests for #5478.

#![cfg(feature = "std")]

// The `Module` derive expands to `::burn::...` paths.
use burn_core as burn;

use burn_core::module::{Module, Param, ParamId};
use burn_core::tensor::{Bool, DType, Device, Int, Tensor, TensorData};
use burn_store::{ApplyError, ModuleSnapshot};

#[derive(Module, Debug)]
struct IntModel {
    ids: Param<Tensor<1, Int>>,
}

#[derive(Module, Debug)]
struct BoolModel {
    mask: Param<Tensor<1, Bool>>,
}

#[derive(Module, Debug)]
struct FloatModel {
    w: Param<Tensor<1>>,
}

fn int_model(device: &Device) -> IntModel {
    IntModel {
        ids: Param::initialized(
            ParamId::new(),
            Tensor::<1, Int>::from_data([1, 2, 3], device),
        ),
    }
}

fn source(data: TensorData, name: &str) -> burn_pack::Tensor {
    burn_store::bridge::from_data(data, name.to_string(), None)
}

/// PyTorch writes integer buffers as int64, and the loaded parameter keeps that width rather
/// than narrowing to the backend's element. Pinned deliberately in #4854; the kind check must
/// not change it.
#[test]
fn an_int_source_keeps_its_own_width() {
    let device: Device = Default::default();
    let mut model = int_model(&device);

    let result = model.apply(
        vec![source(TensorData::from([9i64, 8, 7]), "ids")],
        None,
        None,
        false,
    );

    assert_eq!(result.applied, vec!["ids".to_string()]);
    assert!(result.errors.is_empty(), "{:?}", result.errors);

    let data = model.ids.val().to_data();
    assert_eq!(data.dtype, DType::I64);
    assert_eq!(data.try_to_vec::<i64>().unwrap(), vec![9, 8, 7]);
}

/// Float data in an int parameter is not a width difference, it is the wrong kind. Shape
/// agreement says nothing about it, so before #5478 it loaded and left the parameter F32.
#[test]
fn float_data_is_refused_by_an_int_parameter() {
    let device: Device = Default::default();
    let mut model = int_model(&device);
    let before = model.ids.val().to_data();

    let result = model.apply(
        vec![source(TensorData::from([9.0f32, 8.0, 7.0]), "ids")],
        None,
        None,
        false,
    );

    assert!(result.applied.is_empty(), "{:?}", result.applied);
    assert!(
        matches!(
            result.errors.as_slice(),
            [ApplyError::DTypeMismatch { path, found, .. }]
                if path == "ids" && *found == DType::F32
        ),
        "expected a DTypeMismatch naming the source dtype, got {:?}",
        result.errors
    );

    let after = model.ids.val().to_data();
    assert_eq!(after.dtype, before.dtype, "the parameter was modified");
    assert_eq!(after.try_to_vec::<i32>().unwrap(), vec![1, 2, 3]);
}

/// The mirror case: integer data in a float parameter.
#[test]
fn int_data_is_refused_by_a_float_parameter() {
    let device: Device = Default::default();
    let mut model = FloatModel {
        w: Param::initialized(ParamId::new(), Tensor::<1>::from_data([1.0, 2.0], &device)),
    };

    let result = model.apply(
        vec![source(TensorData::from([7i64, 6]), "w")],
        None,
        None,
        false,
    );

    assert!(result.applied.is_empty());
    assert!(
        matches!(result.errors.as_slice(), [ApplyError::DTypeMismatch { .. }]),
        "got {:?}",
        result.errors
    );
    assert_eq!(
        model.w.val().to_data().try_to_vec::<f32>().unwrap(),
        vec![1.0, 2.0]
    );
}

/// A half-precision save must still load into a full-precision module, so float parameters
/// keep the source's dtype. This is the behaviour `dtype_preservation_f64` already pinned and
/// the dtype check must not break.
#[test]
fn a_float_parameter_still_keeps_the_sources_precision() {
    let device: Device = Default::default();
    let mut model = FloatModel {
        w: Param::initialized(ParamId::new(), Tensor::<1>::from_data([1.0, 2.0], &device)),
    };

    let half = TensorData::from([3.0f32, 4.0]).convert_dtype(DType::F16);
    let result = model.apply(vec![source(half, "w")], None, None, false);

    assert_eq!(result.applied, vec!["w".to_string()]);
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert_eq!(model.w.val().to_data().dtype, DType::F16);
}

/// SafeTensors has no boolean dtype for the non-native bool stores, so it writes `Bool(U32)`
/// as `U32` and reads it back as an integer. A bool parameter has to accept that.
#[test]
fn an_integer_source_still_loads_into_a_bool_parameter() {
    let device: Device = Default::default();
    let mut model = BoolModel {
        mask: Param::initialized(
            ParamId::new(),
            Tensor::<1, Bool>::from_data([false, false, false, false], &device),
        ),
    };

    let result = model.apply(
        vec![source(TensorData::from([1u32, 0, 1, 0]), "mask")],
        None,
        None,
        false,
    );

    assert_eq!(result.applied, vec!["mask".to_string()]);
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    assert_eq!(
        model.mask.val().to_data().try_to_vec::<bool>().unwrap(),
        vec![true, false, true, false]
    );
}

/// Float data in a bool parameter is still the wrong kind.
#[test]
fn float_data_is_refused_by_a_bool_parameter() {
    let device: Device = Default::default();
    let mut model = BoolModel {
        mask: Param::initialized(
            ParamId::new(),
            Tensor::<1, Bool>::from_data([true, true], &device),
        ),
    };

    let result = model.apply(
        vec![source(TensorData::from([1.0f32, 0.0]), "mask")],
        None,
        None,
        false,
    );

    assert!(result.applied.is_empty());
    assert!(
        matches!(result.errors.as_slice(), [ApplyError::DTypeMismatch { .. }]),
        "got {:?}",
        result.errors
    );
}
