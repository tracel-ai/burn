# pytorch-reader

Read PyTorch checkpoint files (`.pt`, `.pth`) without PyTorch or Burn.

[![Current Crates.io Version](https://img.shields.io/crates/v/pytorch-reader.svg)](https://crates.io/crates/pytorch-reader)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

The crate parses the pickle inside a checkpoint and hands back each tensor's name, element
type and shape, with the bytes produced only when asked for. It reads every container
`torch.save` has written: the ZIP archive of PyTorch 1.6 and later, the legacy pickle stream
of 0.1.10 through 1.5, and the TAR archive before that.

```rust
use pytorch_reader::PytorchReader;

let reader = PytorchReader::new("model.pt")?;
for tensor in reader.tensors().values() {
    println!("{}: {:?} {:?}", tensor.name, tensor.dtype(), tensor.shape());
}

// Bytes are read from the file here, not at open.
let weight = reader.get("fc.weight").unwrap();
let bytes: Vec<u8> = weight.read()?;
```

A checkpoint that nests its weights under a key (`"state_dict"`, `"model"`, ...) is opened
with `PytorchReader::with_top_level_key("checkpoint.pt", "state_dict")`. Non-tensor values
(configuration dictionaries, say) can be deserialized into any `serde` type with
`PytorchReader::load_config`.

This is the reader behind [`burn-store`](https://crates.io/crates/burn-store), which wraps
its tensors for loading into [Burn](https://github.com/tracel-ai/burn) modules.
