//! Crate-private macros.
//!
//! [`obfuscate_type`] is the workhorse here: it lets us store a dispatch-level
//! value (e.g. `DispatchDevice`, `BridgeTensorVariant`, `AutodiffGradients`)
//! inside a wrapper type whose own definition does **not** name the inner
//! type. That breaks the chain of type references that would otherwise force
//! downstream crates to monomorphize and resolve the full cubecl-backed type
//! tree just to use a `Device` or `Tensor`.
//!
//! The storage is inline — no heap allocation — and uses an array of
//! pointer-sized cells (`*mut ()`) so that any pointers owned by the inner
//! value keep their Miri/Tree-Borrows provenance through the round trip. A
//! plain `[u8; size_of::<Inner>()]` would strip pointer provenance on read,
//! which manifested as a "dangling pointer with no provenance" UB when the
//! inner value owned an `Arc`/`Box`/etc.
//!
//! Alignment is handled by `#[repr(C, align(64))]` on the blob struct. 64 is
//! overkill for everything we currently store (max alignment is `usize` ⇒ 8
//! on 64-bit targets), but it's also the typical cache-line size and gives
//! us a comfortable safety margin. If a future inner type needs more, the
//! static assert below fires at compile time so the choice is easy to
//! revisit.
//!
//! Auto-traits are opt-in: the macro accepts a trailing list of `Send` /
//! `Sync` tokens, and emits an impl only for each one requested. Earlier
//! versions forwarded these from `$inner` via `where`-bounded `unsafe
//! impl`s, but the bounds were evaluated eagerly on the concrete `$inner`
//! and produced compile errors for types that are `!Sync` (e.g.
//! `AutodiffGradients`, which transitively holds `Box<dyn Any + Send>`).
//! Forcing the caller to opt in keeps the soundness decision visible at the
//! use site.

/// Define a private module that stores a value of `$inner` inside an
/// alignment-correct, type-erased pointer-cell blob.
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
/// The blob carries `#[repr(C, align(64))]`, and the macro emits a `const`
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
/// // No auto-traits — Blob is !Send + !Sync, caller decides.
/// obfuscate_type!(my_blob, MyInner);
///
/// // Opt in to Send + Sync on the generated `Blob`.
/// obfuscate_type!(my_blob, MyInner, Send, Sync);
/// ```
///
/// Only `Send` and `Sync` are recognised in the trailing list; any other
/// identifier is a compile error. Both are emitted as `unsafe impl`s without
/// a bound — opting in is an assertion by the caller that `$inner` actually
/// satisfies the marker.
macro_rules! obfuscate_type {
    ($mod_name:ident, $inner:ty $(, $extra:ident)* $(,)?) => {
        mod $mod_name {
            // Bring the parent's items (incl. `$inner`'s path) into scope.
            #[allow(unused_imports)]
            use super::*;

            const SIZE: usize = ::core::mem::size_of::<$inner>();
            const SLOT: usize = ::core::mem::size_of::<*mut ()>();

            /// Number of pointer-sized cells needed to cover `$inner`. Round
            /// up; the extra bytes past `SIZE` are never read as part of the
            /// inner value.
            const SLOTS: usize = SIZE.div_ceil(SLOT);

            /// The hard cap this module guarantees. Must match the literal in
            /// `#[repr(align(...))]` below — keep them in sync if you change one.
            const MAX_ALIGN: usize = 64;

            // Compile-time check that `$inner` fits the blob's alignment.
            // Without this the casts in `as_ref`/`as_mut`/`Drop` would be UB
            // for inner types with alignment > MAX_ALIGN.
            const _: () = assert!(
                ::core::mem::align_of::<$inner>() <= MAX_ALIGN,
                "obfuscate_type: inner type's alignment exceeds blob alignment (64). \
                 Increase `MAX_ALIGN` and the matching `#[repr(align(...))]`.",
            );

            /// Aligned, opaque storage for one `$inner` value.
            ///
            /// `#[repr(C, align(64))]` guarantees the start of `data` is
            /// 64-byte aligned, which dominates `align_of::<$inner>()` per
            /// the static assertion above. That makes the
            /// `*const _ as *const $inner` cast in the methods sound.
            ///
            /// Cells are typed as pointer-sized `MaybeUninit<*mut ()>` cells
            /// rather than `u8`: the `*mut ()` payload type lets pointers
            /// owned by the inner value retain their provenance through
            /// `ptr::write` → `ptr::read`, while `MaybeUninit` allows the
            /// cells to legally hold uninitialised padding bytes (which
            /// arise for enum types whose variants don't fill the whole
            /// discriminant). Neither wrapper names `$inner`, so the
            /// type-erasure goal is preserved.
            #[repr(C, align(64))]
            pub(super) struct Blob {
                data: [::core::mem::MaybeUninit<*mut ()>; SLOTS],
            }

            // Opt-in trait impls. Each `$extra` token (one of `Send` or
            // `Sync`) is dispatched to the helper macro below, which emits
            // exactly that one impl. Unlisted traits are not implemented —
            // `*mut ()` makes `Blob` `!Send + !Sync` by default.
            $(
                $crate::macros::obfuscate_type_impl!($extra);
            )*

            // The macro exposes a uniform API (`new`/`as_ref`/`as_mut`/
            // `into_inner`) but not every use site needs every entry point —
            // e.g. `Gradients` never moves the inner out, `Device` never
            // mutably borrows. Silence the resulting `dead_code` warnings
            // here rather than at every call site.
            #[allow(dead_code)]
            impl Blob {
                /// Wrap an `$inner` value in a fresh blob.
                pub(super) fn new(inner: $inner) -> Self {
                    let mut blob = Self {
                        data: [::core::mem::MaybeUninit::uninit(); SLOTS],
                    };
                    // SAFETY: `data` is at a 64-byte-aligned offset of `Self`
                    // (which itself has 64-byte alignment from `repr(align)`),
                    // so the cast to `*mut $inner` is properly aligned. The
                    // write covers exactly `SIZE` bytes, which is `<= SLOTS *
                    // SLOT` bytes of valid, exclusive storage.
                    unsafe {
                        (blob.data.as_mut_ptr() as *mut $inner).write(inner);
                    }
                    blob
                }

                /// Borrow the wrapped value.
                pub(super) fn as_ref(&self) -> &$inner {
                    // SAFETY: alignment is guaranteed by `repr(align(64))`
                    // + the `MAX_ALIGN` assert. The bytes were initialized
                    // in `new` and stay initialized until `Drop`/
                    // `into_inner` consume them.
                    unsafe { &*(self.data.as_ptr() as *const $inner) }
                }

                /// Mutably borrow the wrapped value.
                pub(super) fn as_mut(&mut self) -> &mut $inner {
                    // SAFETY: same as `as_ref`; `&mut self` gives exclusive
                    // access.
                    unsafe { &mut *(self.data.as_mut_ptr() as *mut $inner) }
                }

                /// Take ownership of the wrapped value, suppressing `Blob`'s
                /// `Drop` so the inner value's destructor runs exactly once
                /// (when the returned owner is dropped).
                pub(super) fn into_inner(self) -> $inner {
                    // SAFETY: read the bytes through the correctly-typed
                    // pointer, then forget `self` to skip our `Drop` (which
                    // would otherwise drop the value again).
                    let inner: $inner =
                        unsafe { ::core::ptr::read(self.data.as_ptr() as *const $inner) };
                    ::core::mem::forget(self);
                    inner
                }
            }

            impl ::core::ops::Drop for Blob {
                fn drop(&mut self) {
                    // SAFETY: see `as_ref`; running once per `Blob` since
                    // `into_inner` forgets `self` when it consumes the value.
                    unsafe {
                        ::core::ptr::drop_in_place(
                            self.data.as_mut_ptr() as *mut $inner,
                        );
                    }
                }
            }
        }
    };
}

pub(crate) use obfuscate_type;

/// Dispatch a single trait token (one of `Send` or `Sync`) to its
/// corresponding impl on the surrounding module's `Blob` type. Invoked
/// indirectly by `obfuscate_type!` — not intended as a public entry point.
///
/// Unlisted trait names fail to match, which surfaces as a "no rules
/// expected" macro error at the use site — exactly what we want for a typo
/// like `Sned` or for an unsupported trait.
macro_rules! obfuscate_type_impl {
    (Send) => {
        // SAFETY: caller of `obfuscate_type!` asserts that the inner type is
        // safe to move across thread boundaries.
        unsafe impl ::core::marker::Send for Blob {}
    };
    (Sync) => {
        // SAFETY: caller of `obfuscate_type!` asserts that the inner type is
        // safe to share across thread boundaries.
        unsafe impl ::core::marker::Sync for Blob {}
    };
}

pub(crate) use obfuscate_type_impl;

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
