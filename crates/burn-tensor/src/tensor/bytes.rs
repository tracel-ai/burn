//! A version of [`bytemuck::BoxBytes`] that is cloneable and allows trailing uninitialized elements.

use alloc::alloc::{Layout, LayoutError};
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use alloc::vec::Vec;

/// Internally used to avoid accidentally leaking an allocation or using the wrong layout.
struct Allocation {
    /// SAFETY:
    ///  - If `layout.size() > 0`, `ptr` points to a valid allocation from the global allocator
    ///    of the specified layout. The first `len` bytes are initialized.
    ///  - If `layout.size() == 0`, `ptr` is aligned to `layout.align()` and `len` is 0.
    ///    `ptr` is further suitable to be used as the argument for `Vec::from_raw_parts` see [buffer alloc]
    ///    for more details.
    ptr: NonNull<u8>,
    layout: Layout,
}

/// A sort of `Box<[u8]>` that remembers the original alignment and can contain trailing uninitialized bytes.
pub struct Bytes {
    alloc: Allocation,
    // SAFETY: The first `len` bytes of the allocation are initialized
    len: usize,
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
            .field("len", &self.len)
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
        // unwrap here: the layout is valid as it has the alignment & size of self
        Self::try_from_data(self.align(), self.deref()).unwrap()
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl Eq for Bytes {}

impl Allocation {
    // Wrap the allocation of a vector without copying
    fn from_vec<E: Copy>(vec: Vec<E>) -> Self {
        let mut elems = core::mem::ManuallyDrop::new(vec);
        // Set the length to 0, then all data is in the "spare capacity".
        // SAFETY: Data is Copy, so in particular does not need to be dropped. In any case, try not to panic until
        //  we have taken ownership of the data!
        unsafe { elems.set_len(0) };
        let data = elems.spare_capacity_mut();
        // We now have one contiguous slice of data to pass to Layout::for_value.
        let layout = Layout::for_value(data);
        // SAFETY: data is the allocation of a vec, hence can not be null. We use unchecked to avoid a panic-path.
        let ptr = unsafe { NonNull::new_unchecked(elems.as_mut_ptr().cast()) };
        Self { ptr, layout }
    }
    // Create a new allocation with the specified layout
    fn new(layout: Layout) -> Self {
        let ptr = buffer_alloc(layout);
        Self { ptr, layout }
    }
    // Reallocate to fit at least the size and align of min_layout
    fn grow(&mut self, min_layout: Layout) {
        (self.layout, self.ptr) = buffer_grow(self.layout, self.ptr, min_layout);
    }
    // Returns a mutable view of the memory of the whole allocation
    fn memory_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: See type invariants
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.layout.size()) }
    }
    // Return a pointer to the underlying allocation. This pointer is valid for reads and writes until the allocation is dropped or reallocated.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    // Try to convert the allocation to a Vec. The Vec has a length of 0 when returned, but correct capacity and pointer!
    fn try_into_vec<E>(self) -> Result<Vec<E>, Self> {
        let byte_capacity = self.layout.size();
        let Some(capacity) = byte_capacity.checked_div(size_of::<E>()) else {
            return Err(self);
        };
        if capacity * size_of::<E>() != byte_capacity {
            return Err(self);
        };
        if self.layout.align() != align_of::<E>() {
            return Err(self);
        }
        // Okay, let's commit
        let ptr = self.ptr.as_ptr().cast();
        core::mem::forget(self);
        // SAFETY:
        // - ptr was allocated by the global allocator as per type-invariant
        // - `E` has the same alignment as indicated by the stored layout.
        // - capacity * size_of::<E> == layout.size()
        // - 0 <= capacity
        // - no bytes are claimed to be initialized
        // - the layout represents a valid allocation, hence has allocation size less than isize::MAX
        Ok(unsafe { Vec::from_raw_parts(ptr, 0, capacity) })
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        buffer_dealloc(self.layout, self.ptr);
    }
}

// Allocate a pointer that can be passed to Vec::from_raw_parts
fn buffer_alloc(layout: Layout) -> NonNull<u8> {
    // [buffer alloc]: The current docs of Vec::from_raw_parts(ptr, ...) say:
    //   > ptr must have been allocated using the global allocator
    // Yet, an empty Vec is guaranteed to not allocate (it is even illegal! to allocate with a zero-sized layout)
    // Hence, we slightly re-interpret the above to only needing to hold if `capacity > 0`. Still, the pointer
    // must be non-zero. So in case we need a pointer for an empty vec, use a correctly aligned, dangling one.
    if layout.size() == 0 {
        // we would use NonNull:dangling() but we don't have a concrete type for the requested alignment
        let ptr = core::ptr::null_mut::<u8>().wrapping_add(layout.align());
        // SAFETY: layout.align() is never 0
        unsafe { NonNull::new_unchecked(ptr) }
    } else {
        // SAFETY: layout has non-zero size.
        let ptr = unsafe { alloc::alloc::alloc(layout) };
        NonNull::new(ptr).unwrap_or_else(|| alloc::alloc::handle_alloc_error(layout))
    }
}

fn expect_dangling(align: usize, buffer: NonNull<u8>) {
    debug_assert!(
        buffer.as_ptr().wrapping_sub(align).is_null(),
        "expected a nullptr for size 0"
    );
}

#[cold]
fn alloc_overflow() -> ! {
    panic!("Overflow, too many elements")
}

// Grow the buffer while keeping alignment
fn buffer_grow(
    old_layout: Layout,
    buffer: NonNull<u8>,
    min_layout: Layout,
) -> (Layout, NonNull<u8>) {
    let new_align = min_layout.align().max(old_layout.align()); // Don't let data become less aligned
    let new_size = min_layout.size().next_multiple_of(new_align);
    if new_size > isize::MAX as usize {
        alloc_overflow();
    }

    assert!(new_size > old_layout.size(), "size must actually grow");
    if old_layout.size() == 0 {
        expect_dangling(old_layout.align(), buffer);
        let new_layout = Layout::from_size_align(new_size, new_align).unwrap();
        let buffer = buffer_alloc(new_layout);
        return (new_layout, buffer);
    };
    let realloc = || {
        let new_layout = Layout::from_size_align(new_size, old_layout.align()).unwrap();
        // SAFETY:
        // - buffer comes from a Vec or from [`buffer_alloc`/`buffer_grow`].
        // - old_layout is the same as with which the pointer was allocated
        // - new_size is not 0, since it is larger than old_layout.size() which is non-zero
        // - size constitutes a valid layout
        let ptr = unsafe { alloc::alloc::realloc(buffer.as_ptr(), old_layout, new_layout.size()) };
        (new_layout, ptr)
    };
    if new_align == old_layout.align() {
        // happy path. We can just realloc.
        let (new_layout, ptr) = realloc();
        let buffer = NonNull::new(ptr);
        let buffer = buffer.unwrap_or_else(|| alloc::alloc::handle_alloc_error(new_layout));
        return (new_layout, buffer);
    }
    // [buffer grow]: alloc::realloc can *not* change the alignment of the allocation's layout.
    // The unstable Allocator::{grow,shrink} API changes this, but might take a while to make it
    // into alloc::GlobalAlloc.
    //
    // As such, we can not request a specific alignment. But most allocators will give us the required
    // alignment "for free". Hence, we speculatively avoid a mem-copy by using realloc.
    //
    // If in the future requesting an alignment change for an existing is available, this can be removed.
    #[cfg(target_has_atomic = "8")]
    mod alignment_assumption {
        use core::sync::atomic::{AtomicBool, Ordering};
        static SPECULATE: AtomicBool = AtomicBool::new(true);
        pub fn speculate() -> bool {
            // We load and store with relaxed order, since worst case this leads to a few more memcopies
            SPECULATE.load(Ordering::Relaxed)
        }
        pub fn report_violation() {
            SPECULATE.store(false, Ordering::Relaxed)
        }
    }
    #[cfg(not(target_has_atomic = "8"))]
    mod alignment_assumption {
        // On these platforms we don't speculate, and take the hit of performance
        pub fn speculate() -> bool {
            false
        }
        pub fn report_violation() {}
    }
    // reminder: old_layout.align() < new_align
    let mut old_buffer = buffer;
    let mut old_layout = old_layout;
    if alignment_assumption::speculate() {
        let (realloc_layout, ptr) = realloc();
        if let Some(buffer) = NonNull::new(ptr) {
            if buffer.align_offset(new_align) == 0 {
                return (realloc_layout, buffer);
            }
            // Speculating hasn't succeeded, but access now has to go through the reallocated buffer
            alignment_assumption::report_violation();
            old_buffer = buffer;
            old_layout = realloc_layout;
        } else {
            // If realloc fails, the later alloc will likely too, but don't report this yet
        }
    }
    // realloc but change alignment. This requires a mem copy as pointed out above
    let new_layout = Layout::from_size_align(new_size, new_align).unwrap();
    let new_buffer = buffer_alloc(new_layout);
    // SAFETY: two different memory allocations, and old buffer's size is smaller than new_size
    unsafe {
        core::ptr::copy_nonoverlapping(old_buffer.as_ptr(), new_buffer.as_ptr(), old_layout.size());
    }
    buffer_dealloc(old_layout, old_buffer);
    (new_layout, new_buffer)
}

// Deallocate a buffer of a Vec
fn buffer_dealloc(layout: Layout, buffer: NonNull<u8>) {
    if layout.size() != 0 {
        // SAFETY: buffer comes from a Vec or from [`buffer_alloc`/`buffer_grow`].
        // The layout is the same as per type-invariants
        unsafe {
            alloc::alloc::dealloc(buffer.as_ptr(), layout);
        }
    } else {
        // An empty Vec does not allocate, hence nothing to dealloc
        expect_dangling(layout.align(), buffer);
    }
}

impl Bytes {
    /// Copy an existing slice of data into Bytes that are aligned to `align`
    fn try_from_data(align: usize, data: &[u8]) -> Result<Self, LayoutError> {
        let len = data.len();
        let layout = Layout::from_size_align(len, align)?;
        let alloc = Allocation::new(layout);
        unsafe {
            // SAFETY:
            // - data and alloc are distinct allocations of `len` bytes
            core::ptr::copy_nonoverlapping::<u8>(data.as_ref().as_ptr(), alloc.as_mut_ptr(), len);
        };
        Ok(Self { alloc, len })
    }

    /// Ensure the contained buffer is aligned to `align` by possibly moving it to a new buffer.
    fn try_enforce_runtime_align(&mut self, align: usize) -> Result<(), LayoutError> {
        if self.as_mut_ptr().align_offset(align) == 0 {
            // data is already aligned correctly
            return Ok(());
        }
        *self = Self::try_from_data(align, self)?;
        Ok(())
    }

    /// Create a sequence of [Bytes] from the memory representation of an unknown type of elements.
    /// Prefer this over [Self::from_elems] when the datatype is not statically known and erased at runtime.
    pub fn from_bytes_vec(bytes: Vec<u8>) -> Self {
        let mut bytes = Self::from_elems(bytes);
        // TODO: this method could be datatype aware and enforce a less strict alignment.
        // On most platforms, this alignment check is fulfilled either way though, so
        // the benefits of potentially saving a memcopy are negligible.
        bytes.try_enforce_runtime_align(MAX_ALIGN).unwrap();
        bytes
    }

    /// Erase the element type of a vector by converting into a sequence of [Bytes].
    ///
    /// In case the element type is not statically known at runtime, prefer to use [Self::from_bytes_vec].
    pub fn from_elems<E>(elems: Vec<E>) -> Self
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
        let alloc = Allocation::from_vec(elems);
        Self {
            alloc,
            len: byte_len,
        }
    }

    fn reserve(&mut self, additional: usize, align: usize) {
        let needs_to_grow = additional > self.capacity().wrapping_sub(self.len());
        if !needs_to_grow {
            return;
        }
        let Some(required_cap) = self.len().checked_add(additional) else {
            alloc_overflow()
        };
        // guarantee exponential growth for amortization
        let new_cap = required_cap.max(self.capacity() * 2);
        let new_cap = new_cap.max(align); // Small allocations would be pointless
        let Ok(new_layout) = Layout::from_size_align(new_cap, align) else {
            alloc_overflow()
        };
        self.alloc.grow(new_layout);
    }

    /// Extend the byte buffer from a slice of bytes.
    ///
    /// This is used internally to preserve the alignment of the memory layout when matching elements
    /// are extended. Prefer [`Self::extend_from_byte_slice`] otherwise.
    pub(crate) fn extend_from_byte_slice_aligned(&mut self, bytes: &[u8], align: usize) {
        let additional = bytes.len();
        self.reserve(additional, align);
        let len = self.len();
        let new_cap = len.wrapping_add(additional); // Can not overflow, as we've just successfully reserved sufficient space for it
        let uninit_spare = &mut self.alloc.memory_mut()[len..new_cap];
        // SAFETY: reinterpreting the slice as a MaybeUninit<u8>.
        // See also #![feature(maybe_uninit_write_slice)], which would replace this with safe code
        uninit_spare.copy_from_slice(unsafe {
            core::slice::from_raw_parts(bytes.as_ptr().cast(), additional)
        });
        self.len = new_cap;
    }

    /// Extend the byte buffer from a slice of bytes
    pub fn extend_from_byte_slice(&mut self, bytes: &[u8]) {
        self.extend_from_byte_slice_aligned(bytes, MAX_ALIGN)
    }

    /// Get the total capacity, in bytes, of the wrapped allocation.
    pub fn capacity(&self) -> usize {
        self.alloc.layout.size()
    }

    /// Get the alignment of the wrapped allocation.
    pub(crate) fn align(&self) -> usize {
        self.alloc.layout.align()
    }

    /// Convert the bytes back into a vector. This requires that the type has the same alignment as the element
    /// type this [Bytes] was initialized with.
    /// This only returns with Ok(_) if the conversion can be done without a memcopy
    pub fn try_into_vec<E: bytemuck::CheckedBitPattern + bytemuck::NoUninit>(
        mut self,
    ) -> Result<Vec<E>, Self> {
        // See if the length is compatible
        let Ok(data) = bytemuck::checked::try_cast_slice_mut::<_, E>(&mut self) else {
            return Err(self);
        };
        let length = data.len();
        // If so, try to convert the allocation to a vec
        let mut vec = match self.alloc.try_into_vec::<E>() {
            Ok(vec) => vec,
            Err(alloc) => {
                self.alloc = alloc;
                return Err(self);
            }
        };
        // SAFETY: We computed this length from the bytemuck-ed slice into this allocation
        unsafe {
            vec.set_len(length);
        };
        Ok(vec)
    }
}

impl Deref for Bytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: see type invariants
        unsafe { core::slice::from_raw_parts(self.alloc.as_mut_ptr(), self.len) }
    }
}

impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see type invariants
        unsafe { core::slice::from_raw_parts_mut(self.alloc.as_mut_ptr(), self.len) }
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

    #[test]
    fn test_grow() {
        let mut bytes = Bytes::from_elems::<u8>(vec![]);
        bytes.extend_from_byte_slice(&[0, 1, 2, 3]);
        assert_eq!(bytes[..], [0, 1, 2, 3][..]);

        let mut bytes = Bytes::from_elems(vec![42u8; 4]);
        bytes.extend_from_byte_slice(&[0, 1, 2, 3]);
        assert_eq!(bytes[..], [42, 42, 42, 42, 0, 1, 2, 3][..]);
    }

    #[test]
    fn test_large_elems() {
        let mut bytes = Bytes::from_elems(vec![42u128]);
        const TEST_BYTES: [u8; 16] = [
            0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, 0x56,
            0x34, 0x12,
        ];
        bytes.extend_from_byte_slice(&TEST_BYTES);
        let vec = bytes.try_into_vec::<u128>().unwrap();
        assert_eq!(vec, [42u128, u128::from_ne_bytes(TEST_BYTES)]);
    }
}
