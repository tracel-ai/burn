use crate::{element::JitElement, tensor::JitTensor, Runtime};

/// Generates kernel source code by replacing some information using templating.
#[macro_export]
macro_rules! kernel_wgsl {
    (
        $struct:ident,
        $file:expr
    ) => {
        /// Generated kernel from wgsl file.
        #[derive(new)]
        pub struct $struct;

        impl $crate::kernel::StaticKernelSource for $struct {
            fn source() -> $crate::kernel::SourceTemplate {
                $crate::kernel::SourceTemplate::new(include_str!($file))
            }
        }
    };
}

/// Create a vector containing the dimension, strides and shape of tensors.
///
/// # Example
///
/// With two tensors (lhs, rhs)
///
/// | Indexes                  | Value       |
/// |:------------------------:|:-----------:|
/// |           0..1           | D           |
/// |           1..D + 1       | lhs strides |
/// |     (D + 1)..(2 * D + 1) | rhs strides |
/// | (2 * D + 1)..(3 * D + 1) | lhs shape   |
/// | (3 * D + 1)..(4 * D + 1) | rhs shape   |
pub fn build_info<R: Runtime, E: JitElement, const D: usize>(
    tensors: &[&JitTensor<R, E, D>],
) -> Vec<u32> {
    let mut info: Vec<u32> = vec![0; tensors.len() * 2 * D + 1];
    info[0] = D as u32;

    let mut current = 1;
    for tensor in tensors.iter() {
        for d in 0..D {
            info[current] = tensor.strides[d] as u32;
            current += 1;
        }
    }
    for tensor in tensors.iter() {
        for d in 0..D {
            info[current] = tensor.shape.dims[d] as u32;
            current += 1;
        }
    }
    info
}
