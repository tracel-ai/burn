//! A version of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use alloc::alloc::{Layout, LayoutError};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use alloc::vec::Vec;

/// A sort of `Box<[u8]>` that remembers the original alignment and can contain trailing uninitialized bytes.
pub struct Bytes {
    /// SAFETY:
    ///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
    ///    of the specified layout. The first `len` bytes are initialized.
    ///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
    ///    `ptr` is further suitable to be used as the argument for `Vec::from_raw_parts` see [buffer alloc]
    ///    for more details.
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
        #[cold]
        fn too_large<E: serde::de::Error>(len: usize, align: usize) -> E {
            // max_length = largest multiple of align that is <= isize::MAX
            // align is a power of 2, hence a multiple has the lower bits unset. Mask them off to find the largest multiple
            let max_length = (isize::MAX as usize) & !(align - 1);
            E::custom(core::format_args!(
                "length too large: {len}. Expected at most {max_length} bytes"
            ))
        }

        // TODO: we can possibly avoid one copy here by deserializing into an existing, correctly aligned, slice of bytes.
        // We might not be able to predict the length of the data, hence it's far more convenient to let `Vec` handle the growth and re-allocations.
        // Further, on a lot of systems, the allocator naturally aligns data to some reasonably large alignment, where no further copy is then
        // necessary.
        let data: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        // When deserializing, we over-align the data. This saves us from having to encode the alignment (which is platform-dependent in any case).
        // If we had more context information here, we could enforce some (smaller) alignment per data type. But this information is only available
        // in `TensorData`. Moreover it depends on the Deserializer there whether the datatype or data comes first.
        let align = MAX_ALIGN;
        let mut bytes = Self::from_elems(data);
        bytes
            .try_enforce_runtime_align(align)
            .map_err(|_| too_large(bytes.len(), align))?;
        Ok(bytes)
    }
}

impl Clone for Bytes {
    fn clone(&self) -> Self {
        // unwrap here: the layout is always valid as it has the alignment & size of self
        Self::try_from_data(self.layout.align(), self.deref()).unwrap()
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for Bytes {}

// Allocate a pointer that can be passed to Vec::from_raw_parts
fn buffer_alloc(layout: Layout) -> NonNull<u8> {
    // [buffer alloc]: The current docs of Vec::from_raw_parts(ptr, ...) say:
    //   > ptr must have been allocated using the global allocator
    // Yet, an empty Vec is guaranteed to not allocate (it is even illegal! to allocate with a zero-sized layout)
    // Hence, we slightly re-interpret the above to only needing to hold if `capacity > 0`. Still, the pointer
    // must be non-zero. So in case we need a pointer for an empty vec, use a correctly aligned, dangling one.
    if layout.size() == 0 {
        // we would use NonNull:dangling() but we don't have a concrete type for the requested alignment
        let raw = core::ptr::null_mut::<u8>().wrapping_add(layout.align());
        // SAFETY: layout.align() is never 0
        unsafe { NonNull::new_unchecked(raw) }
    } else {
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        NonNull::new(ptr).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout))
    }
}

// Deallocate a buffer of a Vec
fn buffer_dealloc(layout: Layout, buffer: NonNull<u8>) {
    if layout.size() != 0 {
        // SAFETY: buffer comes from a Vec or from [`buffer_alloc`].
        // The layout is the same as per type-invariants
        unsafe {
            alloc::alloc::dealloc(buffer.as_ptr(), layout);
        }
    } else {
        // An empty Vec does not allocate, hence nothing to dealloc
    }
}

impl Bytes {
    /// Copy an existing slice of data into Bytes that are aligned to `align`
    fn try_from_data(align: usize, data: &[u8]) -> Result<Self, LayoutError> {
        let len = data.len();
        let layout = Layout::from_size_align(len, align)?;
        let mem = buffer_alloc(layout);
        unsafe {
            // SAFETY:
            // - data and mem are distinct allocations of `len` bytes
            core::ptr::copy_nonoverlapping::<u8>(data.as_ref().as_ptr(), mem.as_ptr(), len);
        };
        Ok(Self {
            ptr: mem,
            len,
            layout,
        })
    }

    /// Ensure the contained buffer is aligned to `align` by possibly moving it to a new buffer.
    fn try_enforce_runtime_align(&mut self, align: usize) -> Result<(), LayoutError> {
        if self.ptr.align_offset(align) == 0 {
            // data is already aligned correctly
            return Ok(());
        }
        *self = Self::try_from_data(align, self)?;
        Ok(())
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
        // We now have one contiguous slice of data to pass to Layout::for_value.
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
        if capacity * size_of::<E>() != self.layout.size() {
            return Err(self);
        }
        if self.layout.align() != align_of::<E>() {
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
        buffer_dealloc(self.layout, self.ptr);
    }
}

// SAFETY: Bytes behaves like a Box<[u8]> and can contain only elements that are themselves Send
unsafe impl Send for Bytes {}
// SAFETY: Bytes behaves like a Box<[u8]> and can contain only elements that are themselves Sync
unsafe impl Sync for Bytes {}

#[cfg(test)]
mod tests {
    use super::Bytes;
    use alloc::{vec, vec::Vec};

    const _CONST_ASSERTS: fn() = || {
        fn test_send<T: Send>() {}
        fn test_sync<T: Sync>() {}
        test_send::<Bytes>();
        test_sync::<Bytes>();
    };

    fn test_serialization_roundtrip(bytes: &Bytes) {
        let config = bincode::config::standard();
        let serialized =
            bincode::serde::encode_to_vec(bytes, config).expect("serialization to succeed");
        let (roundtripped, _) = bincode::serde::decode_from_slice(&serialized, config)
            .expect("deserialization to succeed");
        assert_eq!(
            bytes, &roundtripped,
            "roundtripping through serialization didn't lead to equal Bytes"
        );
    }

    #[test]
    fn test_serialization() {
        test_serialization_roundtrip(&Bytes::from_elems::<i32>(vec![]));
        test_serialization_roundtrip(&Bytes::from_elems(vec![0xdead, 0xbeaf]));
    }

    #[test]
    fn test_into_vec() {
        // We test an edge case here, where the capacity (but not actual size) makes it impossible to convert to a vec
        let mut bytes = Vec::with_capacity(6);
        let actual_cap = bytes.capacity();
        bytes.extend_from_slice(&[0, 1, 2, 3]);
        let mut bytes = Bytes::from_elems::<u8>(bytes);

        bytes = bytes
            .try_into_vec::<[u8; 0]>()
            .expect_err("Conversion should not succeed for a zero-sized type");
        if actual_cap % 4 != 0 {
            // We most likely get actual_cap == 6, we can't force Vec to actually do that. Code coverage should complain if the actual test misses this
            bytes = bytes.try_into_vec::<[u8; 4]>().err().unwrap_or_else(|| {
                panic!("Conversion should not succeed due to capacity {actual_cap} not fitting a whole number of elements");
            });
        }
        bytes = bytes
            .try_into_vec::<u16>()
            .expect_err("Conversion should not succeed due to mismatched alignment");
        bytes = bytes.try_into_vec::<[u8; 3]>().expect_err(
            "Conversion should not succeed due to size not fitting a whole number of elements",
        );
        let bytes = bytes.try_into_vec::<[u8; 2]>().expect("Conversion should succeed for bit-convertible types of equal alignment and compatible size");
        assert_eq!(bytes, &[[0, 1], [2, 3]]);
    }
}
