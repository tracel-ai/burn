# How to read this book

Throughout this book, we try to keep the following structure

## Linking

When referring to structures or functions within codebase, we provide permalinks to the lines in specific commits, and indicate them by the relative path of their parent file from the project root. For example this is a reference to the `Tensor` struct in [`burn-tensor/src/tensor/api/base.rs`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-tensor/src/tensor/api/base.rs#L23)

When some reference information is useful but is beyond the scope of contributing to burn, we provide that information in a footnote. To build on the previous example, the `Tensor` mentioned is what's referred to as a newtype struct[^1].

Direct hyperlinks are for tools and resources that are not part of the burn project, but are useful for contributing to it. For example, when working on implementing an op for autodiff, it is useful to use [symbolab](https://www.symbolab.com/) to calculate the left and right partial derivatives.

[^1]: for more information on newtype please refer to [the Advanced Types chapter of the Rust Book](https://doc.rust-lang.org/book/ch19-04-advanced-types.html#using-the-newtype-pattern-for-type-safety-and-abstraction)

