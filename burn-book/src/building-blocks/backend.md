# Backend

Nearly everything in Burn is based on the `Backend` trait, which enables you to run tensor operations using different implementations without having to modify your code.
While a backend may not necessarily have autodiff capabilities, the `ADBackend` trait specifies when autodiff is needed.
This trait not only abstracts operations but also tensor, device, and element types, providing each backend the flexibility they need.
It's worth noting that the trait assumes eager mode since burn fully supports dynamic graphs.
However, we may create another API to assist with integrating graph-based backends, without requiring any changes to the user's code.

Users are not expected to directly use the backend trait methods, as it is primarily designed with backend developers in mind rather than Burn users.
Therefore, most Burn userland APIs are generic across backends.
This approach helps users discover the API more organically with proper autocomplete and documentation.
