//! Crate-private macros.
//!
//! [`obfuscate_type`] is the workhorse here: it lets us store a dispatch-level
//! value (e.g. `DispatchDevice`, `BridgeTensorVariant`, `AutodiffGradients`)
//! inside a wrapper type whose own definition does **not** name the inner
//! type. That breaks the chain of type references that would otherwise force
//! downstream crates to monomorphize and resolve the full cubecl-backed type
//! tree just to use a `Device` or `Tensor`.
//!
//! The previous incarnation stored the bytes in a `[u8; size_of::<Inner>()]`
//! field with no alignment marker, which is UB the moment the wrapper is
//! placed in a context that doesn't happen to land at an
//! `align_of::<Inner>()`-aligned address. This macro fixes that by giving the
//! blob `#[repr(align(64))]` and verifying at compile time that the inner
//! type's alignment requirement fits.
//!
//! 64 is overkill for everything we currently store (max alignment is `usize`
//! ⇒ 8 on 64-bit targets), but it's also the typical cache-line size and gives
//! us a comfortable safety margin without measurable code-size impact. If a
//! future inner type needs more, the static assert below fires at compile
//! time so the choice is easy to revisit.

/// Define a private module that stores a value of `$inner` inside an
/// alignment-correct, type-erased byte blob.
///
/// The generated module exposes one public-to-the-parent type, `Blob`, with
/// the following API:
///
/// ```ignore
/// // Construction / destruction
/// Blob::new(inner: $inner) -> Blob
/// Blob::into_inner(self) -> $inner
///
/// // Borrows
/// Blob::as_ref(&self) -> &$inner
/// Blob::as_mut(&mut self) -> &mut $inner
///
/// // Drop runs `$inner`'s destructor exactly once.
/// ```
///
/// The blob carries `#[repr(align(64))]`, and the macro emits a `const`
/// assertion that `align_of::<$inner>() <= 64`. If the assertion fails the
/// crate won't compile — bump both the `repr(align(...))` and the constant
/// here together.
///
/// # Why a macro?
///
/// `size_of::<T>()` and `align_of::<T>()` are usable in `const` position only
/// for concrete types: you cannot write `struct Foo<T> { bytes: [u8; size_of::<T>()] }`
/// without `feature(generic_const_exprs)`. So we expand a fresh, concrete
/// blob type per use site via macro instead.
///
/// # Example
///
/// ```ignore
/// obfuscate_type!(my_blob, MyInner);
/// // …
/// let b = my_blob::Blob::new(MyInner { … });
/// let r: &MyInner = b.as_ref();
/// let v: MyInner = b.into_inner();
/// ```
macro_rules! obfuscate_type {
    ($mod_name:ident, $inner:ty) => {
        mod $mod_name {
            // Bring the parent's items (incl. `$inner`'s path) into scope.
            #[allow(unused_imports)]
            use super::*;

            type Inner = ::core::mem::MaybeUninit<$inner>;

            const SIZE: usize = ::core::mem::size_of::<Inner>();

            /// The hard cap this module guarantees. Must match the literal in
            /// `#[repr(align(...))]` below — keep them in sync if you change one.
            const MAX_ALIGN: usize = 64;

            // Compile-time check that `$inner` fits the blob's alignment.
            // Without this the casts in `as_ref`/`as_mut`/`Drop` would be UB
            // for inner types with alignment > MAX_ALIGN.
            const _: () = assert!(
                ::core::mem::align_of::<Inner>() <= MAX_ALIGN,
                "obfuscate_type: inner type's alignment exceeds blob alignment (64). \
                 Increase `MAX_ALIGN` and the matching `#[repr(align(...))]`.",
            );

            /// Aligned, opaque storage for one `$inner` value.
            ///
            /// `#[repr(align(64))]` on the struct (which contains a single
            /// `[u8; SIZE]` field at offset 0) guarantees the byte array is
            /// 64-byte aligned, which dominates `align_of::<$inner>()` per
            /// the static assertion above. That makes the
            /// `*const _ as *const Inner` cast in the methods sound.
            #[repr(align(64))]
            pub(super) struct Blob {
                bytes: [u8; SIZE],
            }

            // The macro exposes a uniform API (`new`/`as_ref`/`as_mut`/
            // `into_inner`) but not every use site needs every entry point —
            // e.g. `Gradients` never moves the inner out, `Device` never
            // mutably borrows. Silence the resulting `dead_code` warnings
            // here rather than at every call site.
            #[allow(dead_code)]
            impl Blob {
                /// Wrap an `$inner` value in a fresh blob.
                pub(super) fn new(inner: $inner) -> Self {
                    let mut blob = Self { bytes: [0u8; SIZE] };
                    // SAFETY: `bytes` is at offset 0 of a `#[repr(align(64))]`
                    // struct, so its pointer is 64-byte aligned, which covers
                    // any alignment `$inner` needs (checked by `MAX_ALIGN`
                    // assert). The destination is exclusive (just allocated).
                    unsafe {
                        (blob.bytes.as_mut_ptr() as *mut Inner)
                            .write(::core::mem::MaybeUninit::new(inner));
                    }
                    blob
                }

                /// Borrow the wrapped value.
                pub(super) fn as_ref(&self) -> &$inner {
                    // SAFETY: alignment is guaranteed by the struct's
                    // `repr(align(64))` + the MAX_ALIGN assertion. The bytes
                    // were initialized in `new` and stay initialized until
                    // `Drop`/`into_inner` consume them.
                    let inner: &Inner = unsafe { &*(self.bytes.as_ptr() as *const Inner) };
                    unsafe { inner.assume_init_ref() }
                }

                /// Mutably borrow the wrapped value.
                pub(super) fn as_mut(&mut self) -> &mut $inner {
                    // SAFETY: same as `as_ref`; `&mut self` gives exclusive access.
                    let inner: &mut Inner =
                        unsafe { &mut *(self.bytes.as_mut_ptr() as *mut Inner) };
                    unsafe { inner.assume_init_mut() }
                }

                /// Take ownership of the wrapped value, suppressing `Blob`'s
                /// `Drop` so the inner value's destructor runs exactly once
                /// (when the returned owner is dropped).
                pub(super) fn into_inner(self) -> $inner {
                    // SAFETY: read the bytes, then forget `self` to skip
                    // our `Drop` (which would otherwise drop the value again).
                    let inner: Inner =
                        unsafe { ::core::ptr::read(self.bytes.as_ptr() as *const Inner) };
                    ::core::mem::forget(self);
                    unsafe { inner.assume_init() }
                }
            }

            impl Drop for Blob {
                fn drop(&mut self) {
                    // SAFETY: see `as_ref`; running once per `Blob` since
                    // `into_inner` forgets `self` when it consumes the value.
                    let inner: &mut Inner =
                        unsafe { &mut *(self.bytes.as_mut_ptr() as *mut Inner) };
                    unsafe { inner.assume_init_drop() };
                }
            }
        }
    };
}

pub(crate) use obfuscate_type;

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    // ------------------------------------------------------------------
    // 1. Small plain-old-data round-trip.
    // ------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Pod {
        a: u64,
        b: u32,
    }

    obfuscate_type!(pod_blob, Pod);

    #[test]
    fn pod_round_trip() {
        let value = Pod {
            a: 0xDEAD_BEEF,
            b: 7,
        };
        let blob = pod_blob::Blob::new(value.clone());
        assert_eq!(blob.as_ref(), &value);
        assert_eq!(blob.into_inner(), value);
    }

    #[test]
    fn pod_mutation() {
        let mut blob = pod_blob::Blob::new(Pod { a: 0, b: 0 });
        blob.as_mut().a = 99;
        blob.as_mut().b = 1;
        assert_eq!(blob.as_ref(), &Pod { a: 99, b: 1 });
    }

    #[test]
    fn blob_alignment_is_64() {
        // The whole point of the macro: the blob itself has cache-line
        // alignment regardless of what's inside, so the inner pointer
        // cast inside `as_ref`/`as_mut` is sound.
        assert_eq!(core::mem::align_of::<pod_blob::Blob>(), 64);
    }

    // ------------------------------------------------------------------
    // 2. Heap-owning type — verifies Drop runs exactly once via `Drop`
    //    AND that `into_inner` does NOT double-drop.
    // ------------------------------------------------------------------

    struct DropCounter(Arc<AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    obfuscate_type!(counted_blob, DropCounter);

    #[test]
    fn drop_runs_exactly_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let _b = counted_blob::Blob::new(DropCounter(counter.clone()));
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn into_inner_does_not_double_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let blob = counted_blob::Blob::new(DropCounter(counter.clone()));
        let inner = blob.into_inner();
        // The blob is gone but the value lives on in `inner`.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        drop(inner);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // ------------------------------------------------------------------
    // 3. High-alignment inner type — verifies the static assert is
    //    permissive enough for what we actually use it for.
    // ------------------------------------------------------------------

    #[repr(align(32))]
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct HighAlign {
        data: [u64; 4],
    }

    obfuscate_type!(high_align_blob, HighAlign);

    #[test]
    fn high_alignment_inner_works() {
        let value = HighAlign { data: [1, 2, 3, 4] };
        let blob = high_align_blob::Blob::new(value.clone());
        // The wrapped reference must itself be properly aligned for
        // `HighAlign`, otherwise reading any of its fields is UB.
        let r: &HighAlign = blob.as_ref();
        assert_eq!((r as *const HighAlign as usize) % 32, 0);
        assert_eq!(r, &value);
    }

    // ------------------------------------------------------------------
    // 4. Enum with a heap allocation — mirrors the shape of
    //    `DispatchDevice` / `BridgeTensorVariant`.
    // ------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq)]
    #[allow(dead_code)]
    enum Shape {
        Scalar(u64),
        Vector(Vec<u32>),
        Nested(alloc::boxed::Box<Shape>),
    }

    obfuscate_type!(shape_blob, Shape);

    #[test]
    fn enum_with_heap_round_trip() {
        let value = Shape::Nested(alloc::boxed::Box::new(Shape::Vector(alloc::vec![
            1, 2, 3, 4
        ])));
        let blob = shape_blob::Blob::new(value.clone());
        // Mutate, read, then take ownership back.
        let recovered = blob.into_inner();
        assert_eq!(recovered, value);
    }
}
