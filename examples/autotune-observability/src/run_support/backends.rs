/// Selectable backends: (dropdown label, `--backend` value, cargo `--features` value). wgpu is
/// the baseline; the others need their feature (and toolchain), and wgpu can't do tensor cores.
pub(crate) const BACKENDS: [(&str, &str, &str); 5] = [
    ("wgpu", "wgpu", "backend"),
    ("cuda", "cuda", "cuda"),
    ("vulkan", "vulkan", "vulkan"),
    ("metal", "metal", "metal"),
    ("cpu", "cpu", "cpu"),
];
