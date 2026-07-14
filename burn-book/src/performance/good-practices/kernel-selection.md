# Kernel Selection

As mentioned earlier, complex compute-bound operations are highly non-trivial and require many
tricks for optimal performance. However, the way these tricks are applied varies depending on the
hardware and problem shapes. To select the best kernel, we use a search method with a highly
configurable autotune system that performs micro-benchmarks at runtime on the current hardware.

This may trigger a cold start, but the results of these benchmarks are cached on disk for subsequent
executions.

For deployment or training on spot instances, it’s a good idea to bundle the autotune cache with the
code to mitigate cold starts. Refer to the
[CubeCL configuration documentation](https://burn.dev/books/cubecl/advanced-usage/config.html) for
more details on fine-grained settings .

From the user’s point of view, kernel selection shouldn’t be a problem, but as usual, crafting
models with even shapes, multiples of 8, can significantly improve performance. Avoid creating
tensors with shapes that are multiples of 10, like `[1000, 1000]`, as these typically require bounds
checking and may limit vectorization.

Prefer shapes like `[1024, 1024]`, where dimensions are multiples of 32 or powers of 2, as these are
generally optimal. If you have no choice but to use a suboptimal shape, prefer handling it in a
single kernel, transforming it into an optimal shape. It’s better to have a slow neural network
layer followed by fast ones than to propagate unevenness and end up with smaller, but slower,
layers.
