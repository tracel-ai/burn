#![cfg_attr(not(feature = "std"), no_std)]

pub use burn_core::*;

#[cfg(feature = "train")]
pub mod train {
    pub use burn_train::*;
}
