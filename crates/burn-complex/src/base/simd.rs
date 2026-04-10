use crate::base::element::Complex;
use bytemuck::{NoUninit, Pod};
use macerator::{Scalar, VDiv, Simd, Vector, VAdd, VMul, VSub};
use paste::paste;

impl<E> Scalar for Complex<E>
where
    E: Scalar,
{
    type Mask<S: Simd> = E::Mask<S>;
    
    fn lanes<S: Simd>() -> usize {
        E::lanes::<S>()
    }
    
    unsafe fn vload<S: Simd>(ptr: *const Self) -> Vector<S, Self> {
        // Since Complex<E> is repr(C) and consists of two E's, we can 
        // safely transmute a vector of E's into a vector of Complex<E>'s
           unsafe { 
            let vec_e: Vector<S, E> = E::vload(ptr as *const E);
            std::ptr::read(&vec_e as *const Vector<S, E> as *const Vector<S, Self>)
        }
    }
    
    unsafe fn vload_unaligned<S: Simd>(ptr: *const Self) -> Vector<S, Self> {
        todo!()
    }
    
    unsafe fn vload_low<S: Simd>(ptr: *const Self) -> Vector<S, Self> {
        todo!()
    }
    
    unsafe fn vload_high<S: Simd>(ptr: *const Self) -> Vector<S, Self> {
        todo!()
    }
    
    unsafe fn vstore<S: Simd>(ptr: *mut Self, value: Vector<S, Self>) {
        todo!()
    }
    
    unsafe fn vstore_unaligned<S: Simd>(ptr: *mut Self, value: Vector<S, Self>) {
        todo!()
    }
    
    unsafe fn vstore_low<S: Simd>(ptr: *mut Self, value: Vector<S, Self>) {
        todo!()
    }
    
    unsafe fn vstore_high<S: Simd>(ptr: *mut Self, value: Vector<S, Self>) {
        todo!()
    }
    
    unsafe fn mask_store_as_bool<S: Simd>(out: *mut bool, mask: macerator::Mask<S, Self>) {
        todo!()
    }
    
    fn mask_from_bools<S: Simd>(bools: &[bool]) -> macerator::Mask<S, Self> {
        todo!()
    }
    
    fn splat<S: Simd>(self) -> Vector<S, Self> {
        todo!()
    }
}