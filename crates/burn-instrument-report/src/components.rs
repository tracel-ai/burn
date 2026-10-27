//! How a report reads a run's components and the kernels they launched.

use burn_instrument::Kernel;

/// A component's kernels under the shortest names that still tell them apart.
///
/// A kernel's type name carries its whole generic instantiation and runs to
/// several hundred characters, so the table would be unreadable at full
/// length. `Short` — the last path segment before the generics — is what reads
/// well, but three matmul families instantiate one `MatmulEntry` and collapsing
/// them would print two different kernels under one name, which is worse than
/// a long name. So `Short` is used only where it stays unambiguous *within the
/// component*, and the fuller form serves the rest.
pub(crate) fn named(kernels: &[Kernel]) -> Vec<(String, &Kernel)> {
    use cubecl::config::{TypeNameFormatLevel, type_name_format};

    let short: Vec<String> = kernels
        .iter()
        .map(|kernel| type_name_format(&kernel.name, TypeNameFormatLevel::Short))
        .collect();
    let distinct = short.iter().collect::<std::collections::HashSet<_>>().len();
    let level = match distinct == kernels.len() {
        true => TypeNameFormatLevel::Short,
        false => TypeNameFormatLevel::Balanced,
    };
    kernels
        .iter()
        .map(|kernel| (type_name_format(&kernel.name, level), kernel))
        .collect()
}
