#[cfg(feature = "tch")]
mod tch {
    use crate::{BoolVisionOps, FloatVisionOps, IntVisionOps, VisionBackend};
    use burn_tch::{FloatTchElement, LibTorch};

    impl<E: FloatTchElement> BoolVisionOps for LibTorch<E> {}
    impl<E: FloatTchElement> IntVisionOps for LibTorch<E> {}
    impl<E: FloatTchElement> FloatVisionOps for LibTorch<E> {}
    impl<E: FloatTchElement> QVisionOps for LibTorch<E> {}
    impl<E: FloatTchElement> VisionBackend for LibTorch<E> {}
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
