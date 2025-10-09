// This is a temporary file to help with the refactor
// I'll copy the updated from_initializer method here first

    /// Create an Argument and TensorData from an initializer
    /// Returns (Argument with type info, TensorData with actual values)
    pub fn from_initializer(initializer: &TensorProto) -> (Argument, TensorData) {
        let name = initializer.name.clone();

        // 1) Canonical path first.
        match TensorData::try_from(initializer.clone()) {
            Ok(td) => {
                let arg = if td.shape.is_empty() {
                    // rank-0 (scalar)
                    Self {
                        name,
                        ty: ArgType::Scalar(td.elem_type()),
                    }
                } else {
                    Self {
                        name,
                        ty: ArgType::Tensor(TensorType {
                            elem_type: td.elem_type(),
                            rank: td.shape.len(),
                            static_shape: Some(td.shape.clone()),
                        }),
                    }
                };
                (arg, td)
            }
            Err(orig_err) => {
                // 2) Fallback handling for scalars & empty tensors, with precise diagnostics.
                let dims: Vec<i64> = initializer.dims.clone();
                if dims.iter().any(|&d| d < 0) {
                    panic!(
                        "invalid tensor shape (negative dims) for initializer '{}': {:?}",
                        name, dims
                    );
                }

                // Element count implied by dims (treat [] as scalar => 1).
                let dim_elems: usize = if dims.is_empty() {
                    1
                } else {
                    dims.iter().map(|&d| d as usize).product()
                };

                // Payload len across typed fields (best-effort).
                let payload_len = {
                    let i32n = initializer.int32_data.len();
                    let i64n = initializer.int64_data.len();
                    let f32n = initializer.float_data.len();
                    let f64n = initializer.double_data.len();
                    let sn = initializer.string_data.len();
                    let typed = *[i32n, i64n, f32n, f64n, sn].iter().max().unwrap_or(&0);
                    if typed > 0 {
                        typed
                    } else {
                        // raw_data fallback: many exporters put single scalars here
                        if !initializer.raw_data.is_empty() && dim_elems == 1 {
                            1
                        } else {
                            0
                        }
                    }
                };

                // 2.a) Accept scalar encodings: [] or [1] with one element.
                let looks_scalar = dims.is_empty() || (dims.len() == 1 && dims[0] == 1);
                if looks_scalar && payload_len == 1 {
                    let td = TensorData::try_from(initializer.clone()).unwrap_or_else(|_| {
                        panic!(
                            "failed to decode scalar initializer '{}': dims={:?}",
                            name, dims
                        )
                    });
                    let arg = Self {
                        name,
                        ty: ArgType::Scalar(td.elem_type()),
                    };
                    return (arg, td);
                }

                // 2.b) Accept EMPTY tensors: dim_elems == 0 with payload_len == 0.
                if dim_elems == 0 && payload_len == 0 && !dims.is_empty() {
                    // Map ONNX data_type -> ElementType.
                    // (Covers common types used in initializers; extend as needed.)
                    let elem = match initializer.data_type {
                        1 => ElementType::Float32,  // FLOAT
                        2 => ElementType::Uint8,    // UINT8
                        3 => ElementType::Int8,     // INT8
                        4 => ElementType::Uint16,   // UINT16
                        6 => ElementType::Int32,    // INT32
                        7 => ElementType::Int64,    // INT64
                        9 => ElementType::Bool,     // BOOL
                        10 => ElementType::Float16, // FLOAT16
                        11 => ElementType::Float64, // DOUBLE
                        8 => ElementType::String,   // STRING (rare as tensor; empty ok)
                        // If you need more (e.g., UINT32/UINT64), add them here.
                        other => panic!(
                            "unsupported empty-tensor data_type={} for '{}'",
                            other, name
                        ),
                    };

                    // Build empty Data variant corresponding to elem type.
                    let data = match elem {
                        ElementType::Float32 => Data::Float32s(Vec::new()),
                        ElementType::Float64 => Data::Float64s(Vec::new()),
                        ElementType::Float16 => Data::Float16s(Vec::new()),
                        ElementType::Int32 => Data::Int32s(Vec::new()),
                        ElementType::Int64 => Data::Int64s(Vec::new()),
                        ElementType::Uint16 => Data::Uint16s(Vec::new()),
                        ElementType::Uint8 => Data::Uint8s(Vec::new()),
                        ElementType::Int8 => Data::Int8s(Vec::new()),
                        ElementType::Bool => Data::Bools(Vec::new()),
                        ElementType::String => Data::Strings(Vec::new()),
                    };

                    let shape_usize: Vec<usize> = dims.iter().map(|&d| d as usize).collect();

                    let arg = Self {
                        name,
                        ty: ArgType::Tensor(TensorType {
                            elem_type: elem,
                            rank: shape_usize.len(),
                            static_shape: Some(shape_usize.clone()),
                        }),
                    };
                    let td = TensorData {
                        data,
                        shape: shape_usize,
                    };
                    return (arg, td);
                }

                // Not scalar, not empty-tensor; fail with context.
                panic!(
                    "invalid tensor '{}' (dims {:?} => {} elems) with payload {} elems; original error: {:?}",
                    name, dims, dim_elems, payload_len, orig_err
                );
            }
        }
    }
