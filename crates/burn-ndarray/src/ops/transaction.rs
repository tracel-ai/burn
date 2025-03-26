use crate::{
    FloatNdArrayElement, NdArray,
    element::{IntNdArrayElement, QuantElement},
};
use burn_tensor::ops::TransactionOps;

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> TransactionOps<Self>
    for NdArray<E, I, Q>
{
}
