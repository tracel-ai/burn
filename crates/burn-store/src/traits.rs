use alloc::boxed::Box;
use alloc::vec::Vec;

use super::applier::{Applier, ApplyResult};
use crate::collector::Collector;
use crate::{ModuleAdapter, PathFilter, TensorSnapshot};
use burn_core::module::Module;
use burn_tensor::backend::Backend;

/// Extension trait for modules that provides tensor storage functionality.
///
/// This trait provides convenient methods to collect and apply tensor snapshots from any Burn module.
/// Collection operations create lightweight tensor snapshots without immediately copying data.
/// Apply operations apply tensor data from snapshots to the corresponding tensors in the module.
pub trait ModuleSnapshot<B: Backend>: Module<B> {
    /// Collects tensor snapshots for inspection without copying data.
    ///
    /// Returns a vector of `TensorSnapshot` objects that can lazily materialize the tensor data.
    /// Each `TensorSnapshot` contains the full path accessible via `snapshot.full_path()`.
    ///
    /// # Arguments
    ///
    /// * `filter` - An optional [`PathFilter`] to determine which tensors to collect.
    ///   When `None`, all tensors are collected.
    /// * `adapter` - Optional adapter to transform tensors based on container types.
    ///   Applied to all collected tensors before returning.
    fn collect(
        &self,
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
    ) -> Vec<TensorSnapshot> {
        let mut collector = Collector::new(filter, adapter);
        self.visit(&mut collector);
        collector.into_tensors()
    }

    /// Applies tensor snapshots to the module.
    ///
    /// This is the primary apply method that applies tensor data from `TensorSnapshot`s
    /// to the corresponding tensors in the module. The snapshots are typically obtained
    /// from `collect()` or loaded from storage.
    ///
    /// # Arguments
    ///
    /// * `snapshots` - A vector of TensorSnapshot objects
    /// * `filter` - An optional [`PathFilter`] to determine which tensors to apply.
    ///   When `None`, all available tensors are applied.
    /// * `adapter` - Optional adapter to transform tensors based on container types
    ///
    /// # Returns
    ///
    /// An [`ApplyResult`] containing information about applied, skipped, missing,
    /// and unused tensors, as well as any errors encountered.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn_store::PathFilter;
    ///
    /// // Apply all tensors
    /// let result = model.apply(snapshots, None, None);
    ///
    /// // Apply only encoder tensors
    /// let filter = PathFilter::new().with_regex(r"^encoder\..*");
    /// let result = model.apply(snapshots, Some(filter), None);
    ///
    /// // Apply with complex filter
    /// let filter = PathFilter::new()
    ///     .with_regex(r"^encoder\..*")
    ///     .with_regex(r"^decoder\..*")
    ///     .with_full_path("head.weight");
    /// let result = model.apply(snapshots, Some(filter), None);
    /// ```
    fn apply(
        &mut self,
        snapshots: Vec<TensorSnapshot>,
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
    ) -> ApplyResult
    where
        Self: Sized,
    {
        let mut applier = Applier::new(snapshots, filter, adapter);

        // Use unsafe to avoid cloning the entire module, which would double the memory usage
        // We read the module out, map it, then write it back
        // See https://github.com/tracel-ai/burn/issues/3754
        unsafe {
            // Read the module out of self (moves it, leaving self in undefined state)
            let module = core::ptr::read(self as *const Self);

            // Map the module to create a new one with updated tensors
            let new_module = module.map(&mut applier);

            // Write the new module back to self
            core::ptr::write(self as *mut Self, new_module);
        }

        applier.into_result()
    }

    /// Collects tensor snapshots into a [`ModuleSnapshoter`] for saving.
    ///
    /// This method allows using a `ModuleSnapshoter` implementation to handle the
    /// collection and writing logic in a configurable way.
    ///
    /// # Arguments
    ///
    /// * `store` - A mutable reference to a [`ModuleSnapshoter`] that will collect and save the tensors
    fn collect_to<P>(&self, store: &mut P) -> Result<(), P::Error>
    where
        P: ModuleSnapshoter,
    {
        store.collect_from(self)
    }

    /// Applies tensor data from a [`ModuleSnapshoter`] for loading.
    ///
    /// This method allows using a `ModuleSnapshoter` implementation to handle the
    /// loading and application logic in a configurable way.
    ///
    /// # Arguments
    ///
    /// * `store` - A mutable reference to a [`ModuleSnapshoter`] that will load and apply tensors
    fn apply_from<P>(&mut self, store: &mut P) -> Result<ApplyResult, P::Error>
    where
        P: ModuleSnapshoter,
    {
        store.apply_to(self)
    }
}

/// A trait for handling module storage operations.
///
/// `ModuleSnapshoter` provides a unified interface for saving and loading module
/// tensor data with support for various storage formats and advanced features like filtering,
/// remapping, and metadata handling.
pub trait ModuleSnapshoter {
    /// The error type that can be returned during storage operations.
    ///
    /// This should be a format-specific error type that provides detailed
    /// information about what went wrong (e.g., I/O errors, format violations,
    /// unsupported tensor types).
    type Error: core::fmt::Debug + core::fmt::Display;

    /// Collect tensor data from a module and store it to storage.
    ///
    /// This method traverses the module structure, collects all tensor data
    /// according to the store's configuration (filters, remapping, etc.),
    /// and writes it to the underlying storage.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to collect tensor data from. The module must
    ///   implement `ModuleSnapshot` to provide tensor access.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If all tensors were successfully collected and stored
    /// * `Err(Self::Error)` - If an error occurred during collection or writing
    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error>;

    /// Load stored tensor data and apply it to a module.
    ///
    /// This method reads tensor data from storage and applies it to the provided
    /// module. The operation is flexible and can handle partial matches, missing
    /// tensors, and extra tensors in the storage.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to apply tensor data to. The module must
    ///   implement `ModuleSnapshot` to allow tensor updates.
    ///
    /// # Returns
    ///
    /// * `Ok(ApplyResult)` - Detailed information about the apply operation:
    ///   - `applied`: List of successfully applied tensor names
    ///   - `missing`: Tensors expected by the module but not found in storage
    ///   - `skipped`: Tensors in storage that were not applied (filtered or not needed)
    ///   - `errors`: Non-critical errors that occurred during apply
    /// * `Err(Self::Error)` - If a critical error prevented the apply operation
    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error>;
}

// Blanket implementation for all modules
impl<B: Backend, M: Module<B>> ModuleSnapshot<B> for M {}
