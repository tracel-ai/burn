# Custom Kernels

In this section we will cover how to create custom kernels for a backend.
This is especially useful if you want to perform manual optimization on a specific platform while still leveraging the burn ecosystem.
It also give the possibility to call operations that are not yet available on Burn for a backend that exposes that function.

Burn allows you to extend existing backend in a very rusty way.
Crossing the backend abstraction boundaries is possible by creating your own backend trait with the additional operations that you require.

```rust, ignore
trait MyBackend: Backend {
}
```
