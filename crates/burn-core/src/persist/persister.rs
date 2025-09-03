use crate::persist::ModulePersist;
use crate::persist::appliers::ApplyResult;
use crate::tensor::backend::Backend;

/// A trait for handling module persistence operations.
///
/// `ModulePersister` provides a unified interface for saving and loading module
/// tensor data with support for various storage formats and advanced features like filtering,
/// remapping, and metadata handling.
pub trait ModulePersister {
    /// The error type that can be returned during persistence operations.
    ///
    /// This should be a format-specific error type that provides detailed
    /// information about what went wrong (e.g., I/O errors, format violations,
    /// unsupported tensor types).
    type Error: core::fmt::Debug + core::fmt::Display;

    /// Collect tensor data from a module and persist it to storage.
    ///
    /// This method traverses the module structure, collects all tensor data
    /// according to the persister's configuration (filters, remapping, etc.),
    /// and writes it to the underlying storage.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to collect tensor data from. The module must
    ///   implement `ModulePersist` to provide tensor access.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If all tensors were successfully collected and persisted
    /// * `Err(Self::Error)` - If an error occurred during collection or writing
    fn collect_from<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error>;

    /// Load persisted tensor data and apply it to a module.
    ///
    /// This method reads tensor data from storage and applies it to the provided
    /// module. The operation is flexible and can handle partial matches, missing
    /// tensors, and extra tensors in the storage.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to apply tensor data to. The module must
    ///   implement `ModulePersist` to allow tensor updates.
    ///
    /// # Returns
    ///
    /// * `Ok(ApplyResult)` - Detailed information about the apply operation:
    ///   - `applied`: List of successfully applied tensor names
    ///   - `missing`: Tensors expected by the module but not found in storage
    ///   - `skipped`: Tensors in storage that were not applied (filtered or not needed)
    ///   - `errors`: Non-critical errors that occurred during apply
    /// * `Err(Self::Error)` - If a critical error prevented the apply operation
    fn apply_to<B: Backend, M: ModulePersist<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error>;
}
