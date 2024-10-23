//! A verion of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use core::{
    alloc::{Layout, LayoutError},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};
use std::borrow::Cow;

/// A sort of `Box<[u8]>` that remembers the original alignment and can contain trailing uninitialized bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bytes {
    /// SAFETY:
    ///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
    ///    of the specified layout. The first `len` bytes are initialized.
    ///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WireFormat<'a> {
    align: usize,
    #[serde(with = "serde_bytes", borrow)]
    data: Cow<'a, [u8]>,
}

impl serde::Serialize for Bytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        WireFormat {
            align: self.layout.align(),
            data: Cow::Borrowed(self),
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Bytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let wire = WireFormat::deserialize(deserializer)?;
        Self::from_data(wire.align, wire.data)
            .map_err(|_| Error::custom("alignment is invalid, or length too large"))
    }
}

impl Bytes {
    /// Convert from possibly owned data to Bytes.
    fn from_data(align: usize, data: Cow<'_, [u8]>) -> Result<Self, LayoutError> {
        let len = data.len();
        let layout = Layout::from_size_align(len, align)?;
        // TODO: we can possibly avoid a copy here (or even earlier by replacing serde_bytes::deserialize) by deserializing into an existing,
        // correctly aligned, slice of bytes. Since we might not be able to fully predict the length and align ahead of time, this does currently
        // not seem worth the hazzle.
        let bytes = unsafe {
            let mem = std::alloc::alloc(layout);
            std::ptr::copy_nonoverlapping(data.as_ref().as_ptr(), mem, len);
            NonNull::new_unchecked(mem)
        };
        Ok(Self {
            ptr: bytes,
            len,
            layout,
        })
    }

    /// Erase the element type of a vector by converting into a sequence of [Bytes].
    pub fn from_elems<E: bytemuck::NoUninit>(mut elems: Vec<E>) -> Self {
        let _: () = const {
            assert!(
                !std::mem::needs_drop::<E>(),
                "elements must not need a drop impl"
            );
        };
        // Note: going through a Box as in Vec::into_boxed_slice would re-allocate on excess capacity. Avoid that.
        let byte_len = elems.len() * std::mem::size_of::<E>();
        // Set the length to 0, then all data is in the "spare capacity".
        // SAFETY: Careful not to panic now, or this leaks our data!
        unsafe { elems.set_len(0) };
        let data = elems.spare_capacity_mut();
        // We do this to get one contiguous slice of data to pass to Layout::for_value.
        let layout = Layout::for_value(data);
        // SAFETY: data is the allocation of a vec, hence can not be null. We use unchecked to avoid a panic-path.
        let ptr = unsafe { NonNull::new_unchecked(data.as_mut_ptr() as *mut u8) };
        // Now we manage the memory manually, forget the vec.
        std::mem::forget(elems);
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
        if self.layout.align() != std::mem::align_of::<E>() {
            return Err(self);
        }
        let Ok(data) = bytemuck::checked::try_cast_slice_mut::<_, E>(&mut self) else {
            return Err(self);
        };
        let length = data.len();
        let data = data.as_mut_ptr();
        std::mem::forget(self);
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
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see type invariants
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_mut(), self.len) }
    }
}

impl Drop for Bytes {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            unsafe {
                std::alloc::dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}
