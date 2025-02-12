# Saving and Loading Models

Saving your trained machine learning model is quite easy, no matter the output format you choose. As
mentioned in the [Record](./building-blocks/record.md) section, different formats are supported to
serialize/deserialize models. By default, we use the `NamedMpkFileRecorder` which uses the
[MessagePack](https://msgpack.org/) binary serialization format with the help of
[rmp_serde](https://docs.rs/rmp-serde/).

```rust, ignore
// Save model in MessagePack format with full precision
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("Should be able to save the model");
```

Note that the file extension is automatically handled by the recorder depending on the one you
choose. Therefore, only the file path and base name should be provided.

Now that you have a trained model saved to your disk, you can easily load it in a similar fashion.

```rust, ignore
// Load model in full precision from MessagePack file
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model = model
    .load_file(model_path, &recorder, device)
    .expect("Should be able to load the model weights from the provided file");
```

**Note:** models can be saved in different output formats, just make sure you are using the correct
recorder type when loading the saved model. Type conversion between different precision settings is
automatically handled, but formats are not interchangeable. A model can be loaded from one format
and saved to another format, just as long as you load it back with the new recorder type afterwards.

## Initialization from Recorded Weights

The most straightforward way to load weights for a module is simply by using the generated method
[load_record](https://burn.dev/docs/burn/module/trait.Module.html#tymethod.load_record). Note that
parameter initialization is lazy, therefore no actual tensor allocation and GPU/CPU kernels are
executed before the module is used. This means that you can use `init(device)` followed by
`load_record(record)` without any meaningful performance cost.

```rust, ignore
// Create a dummy initialized model to save
let device = Default::default();
let model = Model::<MyBackend>::init(&device);

// Save model in MessagePack format with full precision
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("Should be able to save the model");
```

Afterwards, the model can just as easily be loaded from the record saved on disk.

```rust, ignore
// Load model record on the backend's default device
let record: ModelRecord<MyBackend> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
    .load(model_path.into(), &device)
    .expect("Should be able to load the model weights from the provided file");

// Initialize a new model with the loaded record/weights
let model = Model::init(&device).load_record(record);
```

## No Storage, No Problem!

For applications where file storage may not be available (or desired) at runtime, you can use the
`BinBytesRecorder`.

In the previous examples we used a `FileRecorder` based on the MessagePack format, which could be
replaced with [another file recorder](./building-blocks/record.md#recorder) of your choice. To embed
a model as part of your runtime application, first save the model to a binary file with
`BinFileRecorder`.

```rust, ignore
// Save model in binary format with full precision
let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("Should be able to save the model");
```

Then, in your final application, include the model and use the `BinBytesRecorder` to load it.

Embedding the model as part of your application is especially useful for smaller models but not
recommended for very large models as it would significantly increase the binary size as well as
consume a lot more memory at runtime.

```rust, ignore
// Include the model file as a reference to a byte array
static MODEL_BYTES: &[u8] = include_bytes!("path/to/model.bin");

// Load model binary record in full precision
let record = BinBytesRecorder::<FullPrecisionSettings>::default()
    .load(MODEL_BYTES.to_vec(), device)
    .expect("Should be able to load model the model weights from bytes");

// Load that record with the model
model.load_record(record);
```

This example assumes that the model was already created before loading the model record. If instead
you want to skip the random initialization and directly initialize the weights with the provided
record, you could adapt this like the [previous example](#initialization-from-recorded-weights).
