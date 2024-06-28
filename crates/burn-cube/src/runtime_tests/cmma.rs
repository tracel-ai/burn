use crate as burn_cube;
use burn_cube::prelude::*;
use burn_tensor::ElementConversion;
use half::f16;

#[cube(launch)]
pub fn kernel_cmma(output: &mut Array<F32>, input: &Array<F16>) {
    let mut lhs = cmma::Matrix::<F16>::new(
        cmma::MatrixIdent::A,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
    );
    let mut rhs = cmma::Matrix::<F16>::new(
        cmma::MatrixIdent::B,
        16,
        16,
        16,
        cmma::MatrixLayout::ColMajor,
    );
    let mut out = cmma::Matrix::<F32>::new(
        cmma::MatrixIdent::Accumulator,
        16,
        16,
        16,
        cmma::MatrixLayout::Undefined,
    );
    cmma::fill::<F32>(&mut out, F32::new(0.0));
    cmma::load::<F16>(&mut lhs, input, UInt::new(16));
    cmma::load::<F16>(&mut rhs, input, UInt::new(16));

    cmma::execute::<F16, F16, F32, F32>(&lhs, &rhs, &out, &out);

    cmma::store::<F32>(output, &out, UInt::new(16), cmma::MatrixLayout::RowMajor);
}

pub fn test_kernel_with_generics<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let input: Vec<f16> = (0..256).map(|i| i.elem()).collect();

    let input = client.create(f16::as_bytes(&input));
    let output = client.empty(core::mem::size_of::<f32>() * 256);

    kernel_cmma_launch::<R>(
        client.clone(),
        CubeCount::new(1, 1, 1),
        KernelSettings::default().cube_dim(CubeDim::new(16, 16, 1)),
        ArrayHandle::new(&output, 256),
        ArrayHandle::new(&input, 256),
    );

    let actual = client.read(output.binding()).read_sync().unwrap();
    let actual = f32::from_bytes(&actual);
    println!("{:?}", actual);

    panic!("Testing");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;

        #[test]
        fn test_aa() {
            let client = TestRuntime::client(&Default::default());
            burn_cube::runtime_tests::cmma::test_kernel_with_generics::<TestRuntime>(client);
        }
    };
}
