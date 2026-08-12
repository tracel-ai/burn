use core::marker::PhantomData;
use ndarray::{ArrayBase, ArrayViewMut, Dimension, RawArrayViewMut, RawDataMut};

/// Gives parallel tasks mutable access to disjoint elements of the same array.
///
/// Stores a raw view rather than a `&mut`, because handing out a `&mut` retags it as unique and so
/// invalidates every handle already in flight; only the most recent one would stay usable.
pub(crate) struct UnsafeSharedRef<'a, A, D> {
    view: RawArrayViewMut<A, D>,
    _marker: PhantomData<&'a mut A>,
}

unsafe impl<A: Send, D: Sync> Sync for UnsafeSharedRef<'_, A, D> {}

impl<'a, A, D: Dimension> UnsafeSharedRef<'a, A, D> {
    pub fn new<S: RawDataMut<Elem = A>>(data: &'a mut ArrayBase<S, D>) -> Self {
        Self {
            view: data.raw_view_mut(),
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// Every element accessed through the returned view must be disjoint from every element
    /// accessed through any other live view returned by `get`.
    pub unsafe fn get(&self) -> ArrayViewMut<'a, A, D> {
        unsafe { self.view.clone().deref_into_view_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    /// Two handles alive at once, each writing a disjoint element: the pattern
    /// `run_par!` produces when rayon runs closures on separate threads.
    #[test]
    fn handles_stay_valid_while_another_is_alive() {
        let mut output = Array4::<f32>::zeros((2, 1, 2, 2));
        {
            let shared = UnsafeSharedRef::new(&mut output);
            let mut first = unsafe { shared.get() };
            let mut second = unsafe { shared.get() };
            second[(1, 0, 1, 1)] = 2.0;
            first[(0, 0, 0, 0)] = 1.0;
        }
        assert_eq!(output[(0, 0, 0, 0)], 1.0);
        assert_eq!(output[(1, 0, 1, 1)], 2.0);
    }
}
