use core::cell::UnsafeCell;

/// Similar to `SyncUnsafeCell` see [Rust issues](https://github.com/rust-lang/rust/issues/95439).
pub(crate) struct UnsafeSharedRef<'a, T> {
    cell: UnsafeCell<&'a mut T>,
}

unsafe impl<T> Sync for UnsafeSharedRef<'_, T> {}

impl<'a, T> UnsafeSharedRef<'a, T> {
    pub fn new(data: &'a mut T) -> Self {
        Self {
            cell: UnsafeCell::new(data),
        }
    }
    pub unsafe fn get(&self) -> &'a mut T {
        unsafe { core::ptr::read(self.cell.get()) }
    }
}
