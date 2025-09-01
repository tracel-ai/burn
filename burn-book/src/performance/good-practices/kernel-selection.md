# Kernel Selection

As mentionned earlier, complex compute bound operations are highly non-trivial and requires many tricks for optimal performance.
However, the way those tricks are applied changes depending on the hardware and problem shapes.
To select the best kernel, we use a search method with a highly configurabled autotune system that perform micro benchmarks at runtime on the current hardware.

This might trigger cold-start, but the results of those benchmarks are cached on disk for following execution.

For deployment or training on spot instances, it can be a good idea to bundle the autotune cache with the code to mitigate cold starts.
Refer to the cubecl configuration documentation for more details on fined grained settings [LINK].

From the user point of vue, kernel selection shound't be a problem, but as usual crafting model with even shapes, mulple of 8 might really improve performance.
Avoid creating tensors multiple of 10, like `[1000, 1000]` as it normally requires checkbound and may limit vectorization.

Prefer shapes like `[1024,1024]`, multiple of `32` or power of `2` are normally preferred.
If you don't have the choise but using an unoptimal shape, prefer doing it in a single kernel, transforming it into a correct shape.
It's better to have a slow neural networks layers followed by fast ones, then probagating the uneveness and have smaller, but slower layers.
