# Record

Records are how states are saved with Burn. Compared to most other frameworks, Burn has its own
advanced saving mechanism that allows interoperability between backends with minimal possible
runtime errors. There are multiple reasons why Burn decided to create its own saving formats.

First, Rust has [serde](https://serde.rs/), which is an extremely well-developed serialization and
deserialization library that also powers the `safetensors` format developed by Hugging Face. If used
properly, all the validations are done when deserializing, which removes the need to write
validation code. Since modules in Burn are created with configurations, they can't implement
serialization and deserialization. That's why the record system was created: allowing you to save
the state of modules independently of the backend in use extremely fast while still giving you all
the flexibility possible to include any non-serializable field within your module.

**Why not use safetensors?**

[`safetensors`](https://github.com/huggingface/safetensors) uses serde with the JSON file format and
only supports serializing and deserializing tensors. The record system in Burn gives you the
possibility to serialize any type, which is very useful for optimizers that save their state, but
also for any non-standard, cutting-edge modeling needs you may have. Additionally, the record system
performs automatic precision conversion by using Rust types, making it more reliable with fewer
manual manipulations.

It is important to note that the `safetensors` format uses the word _safe_ to distinguish itself
from Pickle, which is vulnerable to Python code injection. On our end, the simple fact that we use
Rust already ensures that no code injection is possible. If your storage mechanism doesn't handle
data corruption, you might prefer a recorder that performs checksum validation (i.e., any recorder
with Gzip compression).

## Recorder

Recorders are independent of the backend and serialize records with precision and a format. Note
that the format can also be in-memory, allowing you to save the records directly into bytes.

| Recorder               | Format                   | Compression |
| ---------------------- | ------------------------ | ----------- |
| DefaultFileRecorder    | File - Named MessagePack | None        |
| NamedMpkFileRecorder   | File - Named MessagePack | None        |
| NamedMpkGzFileRecorder | File - Named MessagePack | Gzip        |
| BinFileRecorder        | File - Binary            | None        |
| BinGzFileRecorder      | File - Binary            | Gzip        |
| JsonGzFileRecorder     | File - Json              | Gzip        |
| PrettyJsonFileRecorder | File - Pretty Json       | Gzip        |
| BinBytesRecorder       | In Memory - Binary       | None        |

Each recorder supports precision settings decoupled from the precision used for training or
inference. These settings allow you to define the floating-point and integer types that will be used
for serialization and deserialization.

| Setting                   | Float Precision | Integer Precision |
| ------------------------- | --------------- | ----------------- |
| `DoublePrecisionSettings` | `f64`           | `i64`             |
| `FullPrecisionSettings`   | `f32`           | `i32`             |
| `HalfPrecisionSettings`   | `f16`           | `i16`             |

Note that when loading a record into a module, the type conversion is automatically handled, so you
can't encounter errors. The only crucial aspect is using the same recorder for both serialization
and deserialization; otherwise, you will encounter loading errors.

**Which recorder should you use?**

- If you want fast serialization and deserialization, choose a recorder without compression. The one
  with the lowest file size without compression is the binary format; otherwise, the named
  MessagePack could be used.
- If you want to save models for storage, you can use compression, but avoid using the binary
  format, as it may not be backward compatible.
- If you want to debug your model's weights, you can use the pretty JSON format.
- If you want to deploy with `no-std`, use the in-memory binary format and include the bytes with
  the compiled code.

For examples on saving and loading records, take a look at
[Saving and Loading Models](../saving-and-loading.md).
