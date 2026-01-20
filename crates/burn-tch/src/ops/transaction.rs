use burn_backend::ops::TransactionOps;

use crate::{LibTorch, TchElement};

impl<E: TchElement> TransactionOps<Self> for LibTorch<E> {}
