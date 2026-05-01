#[cfg(feature = "tch")]
mod tch {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, VisionBackend};
    use burn_tch::{LibTorch, TchElement};

    impl<E: TchElement> BoolVisionOps for LibTorch<E> {}
    impl<E: TchElement> IntVisionOps for LibTorch<E> {}
    impl<E: TchElement> FloatVisionOps for LibTorch<E> {}
    impl<E: TchElement> VisionBackend for LibTorch<E> {}
}

#[cfg(feature = "flex")]
mod flex {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, VisionBackend};
    use burn_flex::Flex;

    impl BoolVisionOps for Flex {}
    impl IntVisionOps for Flex {}
    impl FloatVisionOps for Flex {}
    impl VisionBackend for Flex {}
}
