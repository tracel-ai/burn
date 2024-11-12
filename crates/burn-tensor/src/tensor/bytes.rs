//! A version of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use alloc::alloc::{Layout, LayoutError};
use alloc::borrow::Cow;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use alloc::vec::Vec;

/// A sort of `Box<[u8]>` that remembers the original alignment and can contain trailing uninitialized bytes.
pub struct Bytes {
    /// SAFETY:
    ///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
    ///    of the specified layout. The first `len` bytes are initialized.
    ///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

/// The maximum supported alignment. The limit exists to not have to store alignment when serializing. Instead,
/// the bytes are always over-aligned when deserializing to MAX_ALIGN.
const MAX_ALIGN: usize = core::mem::align_of::<u128>();

fn debug_from_fn<F: Fn(&mut core::fmt::Formatter<'_>) -> core::fmt::Result>(
    f: F,
) -> impl core::fmt::Debug {
    // See also: std::fmt::from_fn
    struct FromFn<F>(F);
    impl<F> core::fmt::Debug for FromFn<F>
    where
        F: Fn(&mut core::fmt::Formatter<'_>) -> core::fmt::Result,
    {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            (self.0)(f)
        }
    }
    FromFn(f)
}

impl core::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let data = &**self;
        let fmt_data = move |f: &mut core::fmt::Formatter<'_>| {
            if data.len() > 3 {
                // There is a nightly API `debug_more_non_exhaustive` which has `finish_non_exhaustive`
                f.debug_list().entries(&data[0..3]).entry(&"...").finish()
            } else {
                f.debug_list().entries(data).finish()
            }
        };
        f.debug_struct("Bytes")
            .field("data", &debug_from_fn(fmt_data))
            .field("layout", &self.layout)
            .finish()
    }
}

impl serde::Serialize for Bytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::serialize(self.deref(), serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Bytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // When deserializing, we over-align the data. This saves us from having to encode the alignment (which is platform-dependent in any case).
        let data: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        Self::try_from_data(MAX_ALIGN, Cow::Owned(data))
            .map_err(|_| serde::de::Error::custom("alignment is invalid, or length too large"))
    }
}

impl Clone for Bytes {
    fn clone(&self) -> Self {
        // unwrap here: the layout is always valid as it has the alignment & size of self
        Self::try_from_data(self.layout.align(), Cow::Borrowed(self.deref())).unwrap()
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for Bytes {}

impl Bytes {
    /// Convert from possibly owned data to Bytes.
    fn try_from_data(align: usize, data: Cow<'_, [u8]>) -> Result<Self, LayoutError> {
        let len = data.len();
        let layout = Layout::from_size_align(len, align)?;
        // TODO: we can possibly avoid a copy here (or even earlier by replacing serde_bytes::deserialize) by deserializing into an existing,
        // correctly aligned, slice of bytes. Since we might not be able to fully predict the length and align ahead of time, this does currently
        // not seem worth the hassle.
        let bytes = unsafe {
            let mem = alloc::alloc::alloc(layout);
            core::ptr::copy_nonoverlapping(data.as_ref().as_ptr(), mem, len);
            NonNull::new_unchecked(mem)
        };
        Ok(Self {
            ptr: bytes,
            len,
            layout,
        })
    }

    /// Erase the element type of a vector by converting into a sequence of [Bytes].
    pub fn from_elems<E>(mut elems: Vec<E>) -> Self
    where
        // NoUninit implies Copy
        E: bytemuck::NoUninit + Send + Sync,
    {
        let _: () = const {
            assert!(
                core::mem::align_of::<E>() <= MAX_ALIGN,
                "element type not supported due to too large alignment"
            );
        };
        // Note: going through a Box as in Vec::into_boxed_slice would re-allocate on excess capacity. Avoid that.
        let byte_len = elems.len() * core::mem::size_of::<E>();
        // Set the length to 0, then all data is in the "spare capacity".
        // SAFETY: Data is Copy, so in particular does not need to be dropped. In any case, try not to panic until
        //  we have taken ownership of the data!
        unsafe { elems.set_len(0) };
        let data = elems.spare_capacity_mut();
        // We do this to get one contiguous slice of data to pass to Layout::for_value.
        let layout = Layout::for_value(data);
        // SAFETY: data is the allocation of a vec, hence can not be null. We use unchecked to avoid a panic-path.
        let ptr = unsafe { NonNull::new_unchecked(data.as_mut_ptr().cast()) };
        // Now we manage the memory manually, forget the vec.
        core::mem::forget(elems);
        Self {
            ptr,
            len: byte_len,
            layout,
        }
    }

    /// Get the total capacity, in bytes, of the wrapped allocation.
    pub fn capacity(&self) -> usize {
        self.layout.size()
    }

    /// Convert the bytes back into a vector. This requires that the type has the same alignment as the element
    /// type this [Bytes] was initialized with.
    pub fn try_into_vec<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        mut self,
    ) -> Result<Vec<E>, Self> {
        let Some(capacity) = self.layout.size().checked_div(size_of::<E>()) else {
            return Err(self);
        };
        if self.layout.align() != core::mem::align_of::<E>() {
            return Err(self);
        }
        let Ok(data) = bytemuck::checked::try_cast_slice_mut::<_, E>(&mut self) else {
            return Err(self);
        };
        let length = data.len();
        let data = data.as_mut_ptr();
        core::mem::forget(self);
        // SAFETY:
        // - data was allocated by the global allocator as per type-invariant
        // - `E` has the same alignment as indicated by the stored layout.
        // - capacity * size_of::<E> == layout.size()
        // - len <= capacity, because we have a slice of that length into the allocation.
        // - the first `data.len()` are initialized as per the bytemuck check.
        // - the layout represents a valid allocation, hence has allocation size less than isize::MAX
        Ok(unsafe { Vec::from_raw_parts(data, length, capacity) })
    }
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: see type invariants
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see type invariants
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_mut(), self.len) }
    }
}

impl Drop for Bytes {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            unsafe {
                alloc::alloc::dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

// SAFETY: Bytes behaves like a Box<[u8]> and can contain only elements that are themselves Send
unsafe impl Send for Bytes {}
// SAFETY: Bytes behaves like a Box<[u8]> and can contain only elements that are themselves Sync
unsafe impl Sync for Bytes {}

#[cfg(test)]
mod tests {
    use super::Bytes;

    const _CONST_ASSERTS: fn() = || {
        fn test_send<T: Send>() {}
        fn test_sync<T: Sync>() {}
        test_send::<Bytes>();
        test_sync::<Bytes>();
    };
}
