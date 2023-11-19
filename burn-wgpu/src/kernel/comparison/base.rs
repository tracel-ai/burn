use crate::{
    comparison, comparison_elem, comparison_elem_inplace, comparison_inplace,
    element::WgpuElement,
    kernel::{comparison, comparison_elem, comparison_elem_inplace, comparison_inplace},
    tensor::WgpuTensor,
};
use std::mem;

comparison!(Equal, "==");
comparison!(Greater, ">");
comparison!(GreaterEqual, ">=");
comparison!(Lower, "<");
comparison!(LowerEqual, "<=");

comparison_inplace!(EqualInplace, "==");
comparison_inplace!(GreaterInplace, ">");
comparison_inplace!(GreaterEqualInplace, ">=");
comparison_inplace!(LowerInplace, "<");
comparison_inplace!(LowerEqualInplace, "<=");

comparison_elem!(EqualElem, "==");
comparison_elem!(GreaterElem, ">");
comparison_elem!(GreaterEqualElem, ">=");
comparison_elem!(LowerElem, "<");
comparison_elem!(LowerEqualElem, "<=");

comparison_elem_inplace!(EqualElemInplace, "==");
comparison_elem_inplace!(GreaterElemInplace, ">");
comparison_elem_inplace!(GreaterEqualElemInplace, ">=");
comparison_elem_inplace!(LowerElemInplace, "<");
comparison_elem_inplace!(LowerEqualElemInplace, "<=");

pub fn equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
        return comparison_inplace::<EqualInplace, E, D>(lhs, rhs);
    }
    if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
        return comparison_inplace::<EqualInplace, E, D>(rhs, lhs);
    }

    comparison::<Equal, E, D>(lhs, rhs)
}

pub fn greater<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
        return comparison_inplace::<GreaterInplace, E, D>(lhs, rhs);
    }
    if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
        return comparison_inplace::<LowerInplace, E, D>(rhs, lhs);
    }

    comparison::<Greater, E, D>(lhs, rhs)
}

pub fn greater_equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
        return comparison_inplace::<GreaterEqualInplace, E, D>(lhs, rhs);
    }
    if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
        return comparison_inplace::<LowerEqualInplace, E, D>(rhs, lhs);
    }

    comparison::<GreaterEqual, E, D>(lhs, rhs)
}

pub fn lower<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
        return comparison_inplace::<LowerInplace, E, D>(lhs, rhs);
    }
    if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
        return comparison_inplace::<GreaterInplace, E, D>(rhs, lhs);
    }

    comparison::<Lower, E, D>(lhs, rhs)
}

pub fn lower_equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
        return comparison_inplace::<LowerEqualInplace, E, D>(lhs, rhs);
    }
    if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
        return comparison_inplace::<GreaterEqualInplace, E, D>(rhs, lhs);
    }

    comparison::<LowerEqual, E, D>(lhs, rhs)
}

pub fn equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
        return comparison_elem_inplace::<EqualElemInplace, E, D>(lhs, rhs);
    }

    comparison_elem::<EqualElem, E, D>(lhs, rhs)
}

pub fn greater_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
        return comparison_elem_inplace::<GreaterElemInplace, E, D>(lhs, rhs);
    }

    comparison_elem::<GreaterElem, E, D>(lhs, rhs)
}

pub fn lower_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
        return comparison_elem_inplace::<LowerElemInplace, E, D>(lhs, rhs);
    }

    comparison_elem::<LowerElem, E, D>(lhs, rhs)
}

pub fn greater_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
        return comparison_elem_inplace::<GreaterEqualElemInplace, E, D>(lhs, rhs);
    }

    comparison_elem::<GreaterEqualElem, E, D>(lhs, rhs)
}

pub fn lower_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    if mem::size_of::<E>() == mem::size_of::<u32>() && lhs.can_mut() {
        return comparison_elem_inplace::<LowerEqualElemInplace, E, D>(lhs, rhs);
    }

    comparison_elem::<LowerEqualElem, E, D>(lhs, rhs)
}
