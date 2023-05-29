use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI,
};
use std::marker::PhantomData;

pub struct WGPUBackend<G: GraphicsAPI, F: FloatElement, I: IntElement> {
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}
