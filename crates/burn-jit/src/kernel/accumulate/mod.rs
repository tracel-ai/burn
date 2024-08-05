//! Code for accumulate kernels
//!
//! Accumulate is similar to reduce but the output shape is the same as the input shape.
//! Each element in the output contains the accumulated value up to that point.
mod base;
mod naive;
mod shared;