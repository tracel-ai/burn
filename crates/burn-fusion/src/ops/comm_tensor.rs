use burn_backend::ops::CommunicationTensorOps;

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> CommunicationTensorOps<Self> for Fusion<B> {}
