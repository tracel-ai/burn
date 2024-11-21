use burn_tensor::ops::TransactionOps;

use crate::{LibTorch, QuantElement, TchElement};

impl<E: TchElement, Q: QuantElement> TransactionOps<Self> for LibTorch<E, Q> {}
