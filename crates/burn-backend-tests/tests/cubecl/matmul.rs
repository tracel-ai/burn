use super::*;
use burn_cubecl::kernel::matmul::{
    GemmBackend, gemm_backend, gemm_calls, reset_gemm_calls, set_gemm_backend,
};
use burn_tensor::Tolerance;
use burn_tensor::{DType, Device, Distribution, Shape};
use serial_test::serial;

struct NativeBackendGuard {
    prev: GemmBackend,
}

impl NativeBackendGuard {
    fn enter(target: GemmBackend) -> Self {
        let prev = gemm_backend();
        set_gemm_backend(target);
        reset_gemm_calls();
        Self { prev }
    }
}

impl Drop for NativeBackendGuard {
    fn drop(&mut self) {
        set_gemm_backend(self.prev);
    }
}

// Smallest failing case for https://github.com/tracel-ai/burn/issues/5237
#[test]
#[serial]
fn matmul_double_unit_transposed_rhs_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();

    let m = 1024;
    let n = 64;
    let k = 8;

    // TODO: force the DoubleUnit autotune choice
    let lhs: Tensor<2> = Tensor::ones(Shape::new([m, k]), &device);
    let rhs: Tensor<2> = Tensor::ones(Shape::new([n, k]), &device);

    let lhs_ref = Tensor::<2>::from_data(lhs.to_data(), &device_ref);
    let rhs_ref = Tensor::<2>::from_data(rhs.to_data(), &device_ref);

    let out = lhs.matmul(rhs.transpose());
    let out_ref = lhs_ref.matmul(rhs_ref.transpose());

    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
}

// A `nn::Linear` with `LinearLayout::Col` at batch 1 — the shape reported in
// https://github.com/tracel-ai/burn/issues/5264. The col-major weight makes this a
// `VecMat` problem with a transposed rhs, which selects an entirely different set
// of kernels than the row-major layout does.
#[test]
#[serial]
fn matmul_vec_mat_transposed_rhs_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();

    let (m, k, n) = (1, 128, 512);

    let lhs: Tensor<2> = Tensor::random(Shape::new([m, k]), Distribution::Default, &device);
    let rhs: Tensor<2> = Tensor::random(Shape::new([n, k]), Distribution::Default, &device);

    let lhs_ref = Tensor::<2>::from_data(lhs.to_data(), &device_ref);
    let rhs_ref = Tensor::<2>::from_data(rhs.to_data(), &device_ref);

    let out = lhs.matmul(rhs.transpose());
    let out_ref = lhs_ref.matmul(rhs_ref.transpose());

    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}

// A single-row lhs read out of a `[k, 1]` tensor carries strides `[1, 1]`: both dims
// look contiguous, so the layout is only decidable by noticing that the size-1 dim
// can never be indexed past 0. Getting that wrong makes the autotune key describe a
// layout the kernel does not see, and the plan then contains kernels that are
// illegal for the real operand.
#[test]
#[serial]
fn matmul_transposed_operands_single_row_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();

    let (m, k, n) = (1, 128, 512);

    // Built on the reference device and transferred, so neither operand is read
    // before the matmul consumes it.
    let lhs_ref: Tensor<2> = Tensor::random(Shape::new([k, m]), Distribution::Default, &device_ref);
    let rhs_ref: Tensor<2> = Tensor::random(Shape::new([n, k]), Distribution::Default, &device_ref);

    let lhs = Tensor::<2>::from_data(lhs_ref.to_data(), &device);
    let rhs = Tensor::<2>::from_data(rhs_ref.to_data(), &device);

    let out = lhs.transpose().matmul(rhs.transpose());
    let out_ref = lhs_ref.transpose().matmul(rhs_ref.transpose());

    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}

// Tall matrix matmul with extreme aspect ratio ([N, 3] x [3, 3]) should not
// misclassify as GEMV on CPU or cause worker thread stack overflow.
// https://github.com/tracel-ai/burn/issues/5419
#[cfg(feature = "cpu")]
#[test]
#[serial]
fn matmul_tall_matrix_should_not_overflow_stack() {
    let device = Device::default();

    let n = 5_592_405;
    let lhs: Tensor<2> = Tensor::ones(Shape::new([n, 3]), &device);
    let rhs: Tensor<2> = Tensor::from_floats(
        [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        &device,
    );

    let out = lhs.matmul(rhs);

    // Verify row calculation without materializing the full 5.6M row tensor comparison
    let first_row = out.slice([0..1, 0..3]);
    let expected = Tensor::<2>::from_floats([[1.4, 0.2, 1.0]], &device);

    first_row
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}

// Keep the CPU GEMV path covered at the same problematic scale. Unlike the
// general case above, this shape is intentionally classified as `MatVec`.
#[cfg(feature = "cpu")]
#[test]
#[serial]
fn matmul_tall_matrix_vector_should_not_overflow_stack() {
    let device = Device::default();

    let n = 5_592_405;
    let lhs: Tensor<2> = Tensor::ones(Shape::new([n, 3]), &device);
    let rhs: Tensor<2> = Tensor::from_floats([[0.8], [0.6], [0.0]], &device);

    let out = lhs.matmul(rhs);
    let first_row = out.slice([0..1, 0..1]);
    let expected = Tensor::<2>::from_floats([[1.4]], &device);

    first_row
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}

#[test]
#[serial]
fn matmul_zero_k_contraction_should_zero_output() {
    let device = Device::default();
    for backend in [GemmBackend::CuBlas, GemmBackend::Native] {
        let _guard = NativeBackendGuard::enter(backend);
        let lhs: Tensor<2> = Tensor::zeros(Shape::new([2, 0]), &device);
        let rhs: Tensor<2> = Tensor::zeros(Shape::new([0, 3]), &device);
        let out = lhs.matmul(rhs);
        let expected = Tensor::<2>::zeros(Shape::new([2, 3]), &device);
        out.into_data()
            .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::strict());
    }
}

#[test]
#[serial]
fn matmul_native_gemv_and_gemm_should_match_reference() {
    let device = Device::default();
    let device_ref = ReferenceDevice::new();
    let _guard = NativeBackendGuard::enter(GemmBackend::Native);

    // 1. Small-M NT vec4 GEMV (M <= 16, K % 4 == 0, rhs transposed)
    let lhs_ref: Tensor<2> = Tensor::random(Shape::new([4, 64]), Distribution::Default, &device_ref);
    let rhs_ref: Tensor<2> =
        Tensor::random(Shape::new([37, 64]), Distribution::Default, &device_ref);
    let lhs = Tensor::<2>::from_data(lhs_ref.to_data(), &device);
    let rhs = Tensor::<2>::from_data(rhs_ref.to_data(), &device);
    let out = lhs.matmul(rhs.transpose());
    let out_ref = lhs_ref.matmul(rhs_ref.transpose());
    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-4, 1e-4));
    assert!(gemm_calls().1 >= 1, "expected native vec4 GEMV launch");

    // 2. Small-M general GEMV (M <= 16, unaligned K, NN/TN/TT transpose modes + 3D batch broadcast)
    reset_gemm_calls();
    let lhs_ref: Tensor<3> =
        Tensor::random(Shape::new([2, 19, 5]), Distribution::Default, &device_ref);
    let rhs_ref: Tensor<3> =
        Tensor::random(Shape::new([1, 19, 23]), Distribution::Default, &device_ref);
    let lhs = Tensor::<3>::from_data(lhs_ref.to_data(), &device);
    let rhs = Tensor::<3>::from_data(rhs_ref.to_data(), &device);
    let out = lhs.swap_dims(1, 2).matmul(rhs);
    let out_ref = lhs_ref.swap_dims(1, 2).matmul(rhs_ref);
    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-4, 1e-4));
    assert!(gemm_calls().1 >= 1, "expected native general GEMV launch");

    // 3. Large-M SIMT GEMM (M > 16, partial tiles, all 4 transpose combinations)
    for (t_lhs, t_rhs) in [(false, false), (false, true), (true, false), (true, true)] {
        reset_gemm_calls();
        let (m, k, n) = (70, 35, 67);
        let lhs_shape = if t_lhs { [k, m] } else { [m, k] };
        let rhs_shape = if t_rhs { [n, k] } else { [k, n] };
        let lhs_ref: Tensor<2> =
            Tensor::random(Shape::new(lhs_shape), Distribution::Default, &device_ref);
        let rhs_ref: Tensor<2> =
            Tensor::random(Shape::new(rhs_shape), Distribution::Default, &device_ref);
        let lhs = Tensor::<2>::from_data(lhs_ref.to_data(), &device);
        let rhs = Tensor::<2>::from_data(rhs_ref.to_data(), &device);
        let a = if t_lhs { lhs.transpose() } else { lhs };
        let b = if t_rhs { rhs.transpose() } else { rhs };
        let a_ref = if t_lhs { lhs_ref.transpose() } else { lhs_ref };
        let b_ref = if t_rhs { rhs_ref.transpose() } else { rhs_ref };
        let out = a.matmul(b);
        let out_ref = a_ref.matmul(b_ref);
        out.into_data()
            .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(1e-3, 1e-3));
        assert_eq!(gemm_calls(), (0, 1));
    }

    // 4. Large-M F16 GEMM (dispatches to native WMMA when 16x16x16 cmma is supported, else SIMT)
    let lhs_ref: Tensor<2> =
        Tensor::random(Shape::new([68, 48]), Distribution::Default, &device_ref);
    let rhs_ref: Tensor<2> =
        Tensor::random(Shape::new([52, 48]), Distribution::Default, &device_ref);
    let lhs = Tensor::<2>::from_data(lhs_ref.to_data(), &device).cast(DType::F16);
    let rhs = Tensor::<2>::from_data(rhs_ref.to_data(), &device).cast(DType::F16);
    let _ = lhs.clone().into_data();
    let _ = rhs.clone().into_data();
    reset_gemm_calls();
    let out_f16 = lhs.matmul(rhs.transpose());
    let _ = out_f16.clone().into_data();
    assert_eq!(gemm_calls(), (0, 1));
    let out = out_f16.cast(DType::F32);
    let out_ref = lhs_ref.matmul(rhs_ref.transpose());
    out.into_data()
        .assert_approx_eq::<FloatElem>(&out_ref.into_data(), Tolerance::rel_abs(2e-2, 2e-2));
}

