use burn_backend::ops::CommunicationTensorOps;

use crate::{LibTorch, TchElement};

impl<E: TchElement> CommunicationTensorOps<Self> for LibTorch<E> {}
