use alloc::boxed::Box;
use alloc::vec::Vec;

use super::applier::Applier;
use super::apply_result::ApplyResult;
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
    /// * `skip_enum_variants` - Skip enum variant names when building paths.
    ///   When true, paths will not include enum variant names (e.g., "feature.weight"
    ///   instead of "feature.BaseConv.weight"). Useful when exporting to formats
    ///   like PyTorch/SafeTensors that don't use enum variants.
    fn collect(
        &self,
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
        skip_enum_variants: bool,
    ) -> Vec<TensorSnapshot> {
        let mut collector = Collector::new(filter, adapter, skip_enum_variants);
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
    /// * `skip_enum_variants` - Skip enum variant names when matching tensor paths
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
    /// let result = model.apply(snapshots, None, None, false);
    ///
    /// // Apply only encoder tensors
    /// let filter = PathFilter::new().with_regex(r"^encoder\..*");
    /// let result = model.apply(snapshots, Some(filter), None, false);
    ///
    /// // Apply with complex filter
    /// let filter = PathFilter::new()
    ///     .with_regex(r"^encoder\..*")
    ///     .with_regex(r"^decoder\..*")
    ///     .with_full_path("head.weight");
    /// let result = model.apply(snapshots, Some(filter), None, false);
    ///
    /// // Apply with enum variant skipping (for PyTorch models)
    /// let result = model.apply(snapshots, None, None, true);
    /// ```
    fn apply(
        &mut self,
        snapshots: Vec<TensorSnapshot>,
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
        skip_enum_variants: bool,
    ) -> ApplyResult
    where
        Self: Sized,
    {
        let mut applier = Applier::new(snapshots, filter, adapter, skip_enum_variants);

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

    /// Saves tensor snapshots into a [`ModuleStore`].
    ///
    /// This method allows using a `ModuleStore` implementation to handle the
    /// collection and writing logic in a configurable way.
    ///
    /// # Arguments
    ///
    /// * `store` - A mutable reference to a [`ModuleStore`] that will collect and save the tensors
    fn save_into<P>(&self, store: &mut P) -> Result<(), P::Error>
    where
        P: ModuleStore,
    {
        store.collect_from(self)
    }

    /// Loads tensor data from a [`ModuleStore`].
    ///
    /// This method allows using a `ModuleStore` implementation to handle the
    /// loading and application logic in a configurable way.
    ///
    /// # Arguments
    ///
    /// * `store` - A mutable reference to a [`ModuleStore`] that will load and apply tensors
    fn load_from<P>(&mut self, store: &mut P) -> Result<ApplyResult, P::Error>
    where
        P: ModuleStore,
    {
        store.apply_to(self)
    }
}

/// A trait for handling module storage operations.
///
/// `ModuleStore` provides a unified interface for saving and loading module
/// tensor data with support for various storage formats and advanced features like filtering,
/// remapping, and metadata handling.
pub trait ModuleStore {
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

    /// Get a single tensor snapshot by name.
    ///
    /// This method provides direct access to individual tensors in storage without
    /// requiring a module. The returned `TensorSnapshot` uses lazy loading - tensor
    /// data is only materialized when `to_data()` is called.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name/path (e.g., "encoder.layer1.weight")
    ///
    /// # Returns
    ///
    /// * `Ok(Some(TensorSnapshot))` - The tensor snapshot if found
    /// * `Ok(None)` - If no tensor with that name exists
    /// * `Err(Self::Error)` - If an error occurred accessing storage
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut store = BurnpackStore::from_file("model.bpk");
    /// if let Some(snapshot) = store.get_snapshot("encoder.weight")? {
    ///     println!("Shape: {:?}", snapshot.shape);
    ///     println!("Dtype: {:?}", snapshot.dtype);
    ///     let data = snapshot.to_data()?;  // Lazy load
    /// }
    /// ```
    fn get_snapshot(&mut self, name: &str) -> Result<Option<TensorSnapshot>, Self::Error>;

    /// Get all tensor snapshots from storage.
    ///
    /// This method returns all tensors in storage as lazy-loading snapshots.
    /// Useful for inspection, debugging, or custom tensor manipulation workflows.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TensorSnapshot>)` - All tensor snapshots in storage
    /// * `Err(Self::Error)` - If an error occurred accessing storage
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut store = SafetensorsStore::from_file("model.safetensors");
    /// let snapshots = store.get_snapshots()?;
    /// for snapshot in &snapshots {
    ///     println!("{}: {:?}", snapshot.full_path(), snapshot.shape);
    /// }
    /// ```
    fn get_snapshots(&mut self) -> Result<Vec<TensorSnapshot>, Self::Error>;

    /// Get all tensor names/keys in storage.
    ///
    /// This method returns the names of all tensors without loading any data.
    /// Useful for inspecting storage contents or checking if specific tensors exist.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<String>)` - All tensor names in storage
    /// * `Err(Self::Error)` - If an error occurred accessing storage
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut store = PytorchStore::from_file("model.pth");
    /// let keys = store.keys()?;
    /// println!("Tensors in file: {:?}", keys);
    /// ```
    fn keys(&mut self) -> Result<Vec<String>, Self::Error>;
}

// Blanket implementation for all modules
impl<B: Backend, M: Module<B>> ModuleSnapshot<B> for M {}
