use burn_backend::ops::TransactionOps;

use crate::{LibTorch, TchElement, TchFloatElement, TchIntElement};

impl<E: TchElement, F: TchFloatElement> TransactionOps<Self> for LibTorch<E, F> {}
