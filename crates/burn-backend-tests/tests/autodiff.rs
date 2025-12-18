//! Burn autodiff tests.

pub type FloatElemType = f32;
#[allow(unused)]
pub type IntElemType = i32;

#[allow(clippy::module_inception)]
#[path = "common/autodiff.rs"]
mod autodiff;
