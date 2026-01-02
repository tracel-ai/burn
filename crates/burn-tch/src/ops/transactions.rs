use burn_backend::ops::TransactionOps;

use crate::{LibTorch, TchElement, TchFloatElement, TchIntElement};

impl<E: TchElement, F: TchFloatElement, I: TchIntElement> TransactionOps<Self>
    for LibTorch<E, F, I>
{
}
