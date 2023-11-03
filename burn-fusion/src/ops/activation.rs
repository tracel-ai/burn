use crate::{graph::FusedBackend, FusionBackend};
use burn_tensor::{backend::Backend, ops::ActivationOps};

impl<B: FusedBackend> ActivationOps<Self> for FusionBackend<B>
where
    <B::FullPrecisionBackend as Backend>::FullPrecisionBackend: FusedBackend,
    B::FullPrecisionBackend: FusedBackend,
{
}
