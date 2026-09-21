# Project Architecture

This section documents most major architectural decisions with the reasoning behind them.

**Sections**

- [Module](./module.md)
  - [Optimization](./module.md#optimization)
- [Serialization](./serialization.md)
  - [Constraints](./serialization.md#constraints)
  - [The burnpack format](./serialization.md#the-burnpack-format)
  - [The three record types](./serialization.md#the-three-record-types)
- [Tensor](./tensor.md)
  - [From backend generics to runtime selection](./tensor.md#from-backend-generics-to-runtime-selection)
  - [Why the bridge is opaque](./tensor.md#why-the-bridge-is-opaque)
  - [Following an addition through the stack](./tensor.md#following-an-addition-through-the-stack)
- [Backend](./backend.md)
  - [Autodiff](./backend.md#autodiff)
