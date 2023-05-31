use std::marker::PhantomData;

use crate::element::WGPUElement;

pub trait KernelGenerator {
    type Source: AsRef<str>;

    fn generate() -> Self::Source;
}

#[macro_export]
macro_rules! kernel_wgsl_2 {
    (
        $struct:ident,
        $file:expr
    ) => {
        #[derive(new)]
        pub struct $struct;

        impl KernelGenerator for $struct {
            type Source = &'static str;

            fn generate() -> Self::Source {
                include_str!($file)
            }
        }
    };
}

kernel_wgsl_2!(Test, "../template/binary_elemwise.wgsl");

pub struct KernelTemplate<
    K: KernelGenerator,
    E: WGPUElement,
    I: WGPUElement,
    const WORKGROUP_X_SIZE: usize,
    const WORKGROUP_Y_SIZE: usize,
    const WORKGROUP_Z_SIZE: usize,
> {
    _k: PhantomData<K>,
    _e: PhantomData<E>,
    _i: PhantomData<I>,
}

impl<
        K: KernelGenerator,
        E: WGPUElement,
        I: WGPUElement,
        const WORKGROUP_X_SIZE: usize,
        const WORKGROUP_Y_SIZE: usize,
        const WORKGROUP_Z_SIZE: usize,
    > KernelGenerator
    for KernelTemplate<K, E, I, WORKGROUP_X_SIZE, WORKGROUP_Y_SIZE, WORKGROUP_Z_SIZE>
{
    type Source = String;

    fn generate() -> String {
        let mut source = K::generate().as_ref().to_string();

        source = source.replace("WORKGROUP_SIZE_X", &WORKGROUP_X_SIZE.to_string());
        source = source.replace("WORKGROUP_SIZE_Y", &WORKGROUP_Y_SIZE.to_string());
        source = source.replace("WORKGROUP_SIZE_Z", &WORKGROUP_Y_SIZE.to_string());
        source = source.replace("elem", E::type_name());
        source = source.replace("int", I::type_name());

        source
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::any::TypeId;

    #[test]
    fn test_kernel_type_id() {
        kernel_wgsl_2!(Add, "../template/binary_elemwise.wgsl");

        let type_id_1 = TypeId::of::<KernelTemplate<Add, f32, i32, 2, 3, 4>>();
        let type_id_2 = TypeId::of::<KernelTemplate<Add, f32, i32, 2, 3, 5>>();
        let type_id_3 = TypeId::of::<KernelTemplate<Add, f32, i32, 2, 3, 4>>();

        assert_ne!(type_id_1, type_id_2);
        assert_eq!(type_id_1, type_id_3);
    }
}
