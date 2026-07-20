//! LSTM module.
//!
//! This module implements LSTM cells and variants.
//!
//! # References
//!
//! - [Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
//! - [A Simple Neural Network and Deep Learning Algorithm for Prediction of Stock Prices](https://www.researchgate.net/publication/221444045_A_Simple_Neural_Network_and_Deep_Learning_Algorithm_for_Prediction_of_Stock_Prices)
//! - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
mod bilstm_module;
mod lstm_module;
mod lstm_state;

#[doc(inline)]
pub use bilstm_module::*;
#[doc(inline)]
pub use lstm_module::*;
#[doc(inline)]
pub use lstm_state::*;
