use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};
use burn_tensor::ElementConversion;
use burn_tensor::{ops::TensorOps, Shape};
use ndarray::s;

#[cfg(feature = "std")]
use rayon::prelude::*;

use super::blas::BlasGemm;
use super::matrixmultiply::MatrixmultiplyGemm;

pub(crate) trait Matmul<E> {
    fn matmul<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D>;
}

pub(crate) fn matmul<const D: usize, E: FloatNdArrayElement>(
    lhs: NdArrayTensor<E, D>,
    rhs: NdArrayTensor<E, D>,
) -> NdArrayTensor<E, D> {
    E::matmul(lhs, rhs)
}

impl Matmul<f32> for f32 {
    fn matmul<const D: usize>(
        lhs: NdArrayTensor<f32, D>,
        rhs: NdArrayTensor<f32, D>,
    ) -> NdArrayTensor<f32, D> {
        // matmul_root::<f32, D, MatrixmultiplyGemm>(lhs, rhs)
        matmul_root::<f32, D, BlasGemm>(lhs, rhs)
    }
}

pub struct SharedBuffer<'a, E> {
    cell: core::cell::UnsafeCell<&'a mut [E]>,
}

unsafe impl<'a, E> Sync for SharedBuffer<'a, E> {}

impl<'a, E> SharedBuffer<'a, E> {
    pub fn new(data: &'a mut [E]) -> Self {
        Self {
            cell: core::cell::UnsafeCell::new(data),
        }
    }
    pub unsafe fn get(&self) -> &'a mut [E] {
        unsafe { core::ptr::read(self.cell.get()) }
    }
}

fn matmul_root<E, const D: usize, F>(
    lhs: NdArrayTensor<E, D>,
    rhs: NdArrayTensor<E, D>,
) -> NdArrayTensor<E, D>
where
    E: FloatNdArrayElement,
    F: Gemm<E>,
{
    let shape_ori_lhs = lhs.shape();
    let shape_ori_rhs = rhs.shape();

    let lhs = reshape::<E, D>(lhs);
    let rhs = reshape::<E, D>(rhs);

    let [batch_size_lhs, m, _] = lhs.shape().dims;
    let [batch_size_rhs, k, n] = rhs.shape().dims;

    let mut shape_out = match batch_size_lhs > batch_size_rhs {
        true => shape_ori_lhs,
        false => shape_ori_rhs,
    };
    shape_out.dims[D - 2] = m;
    shape_out.dims[D - 1] = n;

    let out = NdArrayBackend::<E>::empty(shape_out, &NdArrayDevice::Cpu);

    let lhs_strides = lhs.array.strides().to_vec();
    let rhs_strides = rhs.array.strides().to_vec();
    let out_strides = out.array.strides().to_vec();

    run::<E, D, F>(
        m,
        k,
        n,
        lhs,
        lhs_strides,
        rhs,
        rhs_strides,
        out,
        out_strides,
    )
}

pub(crate) trait Gemm<E: FloatNdArrayElement> {
    fn run(
        m: usize,
        k: usize,
        n: usize,
        alpha: E,
        a: *const E,
        rsa: isize,
        csa: isize,
        b: *const E,
        rsb: isize,
        csb: isize,
        beta: E,
        c: *mut E,
        rsc: isize,
        csc: isize,
    );
}

fn run<E: FloatNdArrayElement, const D: usize, Kernel>(
    m: usize,
    k: usize,
    n: usize,
    lhs: NdArrayTensor<E, 3>,
    lhs_strides: Vec<isize>,
    rhs: NdArrayTensor<E, 3>,
    rhs_strides: Vec<isize>,
    mut out: NdArrayTensor<E, D>,
    out_strides: Vec<isize>,
) -> NdArrayTensor<E, D>
where
    Kernel: Gemm<E>,
{
    println!("Lhs {:?}", lhs.shape().dims);
    println!("Rhs {:?}", rhs.shape().dims);

    let run = || {
        let [batch_size_lhs, _, _] = lhs.shape().dims;
        let [batch_size_rhs, _, _] = rhs.shape().dims;
        let batch_size = usize::max(batch_size_rhs, batch_size_lhs);

        if batch_size_lhs > batch_size && batch_size_lhs != 1 {
            panic!("Broadcast on multiple dimensions is not yet supported");
        }

        if batch_size_rhs > batch_size && batch_size_rhs != 1 {
            panic!("Broadcast on multiple dimensions is not yet supported");
        }

        let alpha: E = 1.0.elem();
        let beta: E = 0.0.elem();

        let out_slices = out
            .array
            .as_slice_mut()
            .expect("Data is contiguous and in standard order");

        let buffer = SharedBuffer::new(out_slices);

        // #[cfg(feature = "std")]
        // let iter = (0..batch_size).into_par_iter();
        // #[cfg(not(feature = "std"))]
        let iter = (0..batch_size).into_iter();

        iter.for_each(|b| {
            let lhs_slice = match batch_size_lhs == 1 {
                true => lhs.array.slice(s!(0, .., ..)),
                false => lhs.array.slice(s!(b, .., ..)),
            };
            let rhs_slice = match batch_size_rhs == 1 {
                true => rhs.array.slice(s!(0, .., ..)),
                false => rhs.array.slice(s!(b, .., ..)),
            };

            unsafe {
                let buffer = buffer.get();

                Kernel::run(
                    m,
                    k,
                    n,
                    alpha,
                    lhs_slice.as_ptr(),
                    lhs_strides[1],
                    lhs_strides[2],
                    rhs_slice.as_ptr(),
                    rhs_strides[1],
                    rhs_strides[2],
                    beta,
                    &mut buffer[b * (m * n)],
                    out_strides[D - 2],
                    out_strides[D - 1],
                );
            }
        });

        out
    };
    // #[cfg(feature = "std")]
    // let output = rayon::scope(|_| run());
    // #[cfg(not(feature = "std"))]
    let output = run();

    output
}

impl Matmul<f64> for f64 {
    fn matmul<const D: usize>(
        lhs: NdArrayTensor<f64, D>,
        rhs: NdArrayTensor<f64, D>,
    ) -> NdArrayTensor<f64, D> {
        matmul_root::<f64, D, MatrixmultiplyGemm>(lhs, rhs)
    }
}

fn reshape<E: FloatNdArrayElement, const D: usize>(
    tensor: NdArrayTensor<E, D>,
) -> NdArrayTensor<E, 3> {
    let shape = tensor.shape();

    if D < 2 {
        NdArrayBackend::<E>::reshape(tensor, Shape::new([1, 1, shape.dims[0]]))
    } else {
        let batch_size = batch_size(&shape);
        let size0 = shape.dims[D - 2];
        let size1 = shape.dims[D - 1];

        NdArrayBackend::<E>::reshape(tensor, Shape::new([batch_size, size0, size1]))
    }
}

fn batch_size<const D: usize>(shape: &Shape<D>) -> usize {
    let mut num_batch = 1;
    for i in 0..D - 2 {
        num_batch *= shape.dims[i];
    }

    num_batch
}
