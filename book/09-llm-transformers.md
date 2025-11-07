# Chapter 9: Working with LLMs and Transformers

At the heart of virtually all modern Large Language Models (LLMs) is the **Transformer** architecture. Originally introduced in the paper "Attention Is All You Need," it has become the de facto standard for sequence modeling tasks. This final chapter provides a deep dive into how the Transformer architecture is implemented in Burn, focusing on the `TransformerEncoder` module.

## The Transformer in Burn: A Composite `Module`

The `TransformerEncoder` (found in `crates/burn-nn/src/modules/transformer/encoder.rs`) is a perfect example of Burn's philosophy of building complex models by composing simpler `Module`s.

A `TransformerEncoder` is not a monolithic block; it is a stack of identical layers.

```rust
// crates/burn-nn/src/modules/transformer/encoder.rs

#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    pub layers: Vec<TransformerEncoderLayer<B>>,
    // ... other config fields
}
```

The `TransformerEncoder` itself is a `Module` that contains a `Vec` of `TransformerEncoderLayer`s. The `#[derive(Module)]` macro ensures that all the parameters within this `Vec` of layers are correctly registered, tracked, and serialized.

The `forward` pass of the `TransformerEncoder` is simple: it just iterates through the layers and applies each one in sequence.

```rust
// crates/burn-nn/src/modules/transformer/encoder.rs

impl<B: Backend> TransformerEncoder<B> {
    pub fn forward(&self, input: TransformerEncoderInput<B>) -> Tensor<B, 3> {
        let mut x = input.tensor;

        for layer in self.layers.iter() {
            x = layer.forward(x, input.mask_pad.clone(), input.mask_attn.clone());
        }

        x
    }
}
```

## The `TransformerEncoderLayer`: The Core Building Block

Each `TransformerEncoderLayer` performs the core logic of the transformer. It is, itself, a `Module` composed of smaller, specialized modules.

```rust
// crates/burn-nn/src/modules/transformer/encoder.rs

#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    mha: MultiHeadAttention<B>,
    pwff: PositionWiseFeedForward<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
    dropout: Dropout,
    norm_first: bool,
}
```

### Deeper Dive into the Sub-Modules

*   **`mha: MultiHeadAttention<B>`**: This is the **multi-head self-attention** module. It's the key innovation of the Transformer. For each token in the sequence, it computes three vectors: a **Query**, a **Key**, and a **Value**. It then calculates a score for how much the current token's Query "matches" the Keys of all other tokens in the sequence. These scores are used to create a weighted sum of all the Value vectors, producing an output that is a blend of information from the entire sequence. "Multi-head" means that this process is done multiple times in parallel with different sets of learned weights, and the results are concatenated. This allows the model to focus on different aspects of the sequence simultaneously.

*   **`pwff: PositionWiseFeedForward<B>`**: This is a **position-wise feed-forward network**. It's a small, two-layer neural network (typically `Linear -> ReLU -> Linear`) that is applied independently to each position in the sequence after the attention step. It can be thought of as a "processing" step that adds more capacity to the model.

*   **`norm_1: LayerNorm<B>`** and **`norm_2: LayerNorm<B>`**: These are **layer normalization** modules. They are used to stabilize the training of deep networks by normalizing the outputs of the sub-layers, ensuring that the values don't become too large or too small.

### The Data Flow Within a Layer

The `forward` pass of a `TransformerEncoderLayer` defines the flow of data through these sub-modules. The process involves two main "residual" blocks.

#### Block 1: Multi-Head Attention

1.  **Normalization (Optional)**: If `norm_first` is true, the input is first passed through `norm_2`.
2.  **Self-Attention**: The (potentially normalized) input is passed to the `MultiHeadAttention` module (`mha`).
3.  **Dropout**: Dropout is applied to the output of the attention module.
4.  **Residual Connection**: The output of the dropout layer is added back to the *original* input of the block. This is the "residual connection" or "skip connection," a crucial technique for training deep networks.

#### Block 2: Feed-Forward Network

1.  **Normalization**: The output of the first block is passed through `norm_1`.
2.  **Feed-Forward**: The normalized output is passed through the `PositionWiseFeedForward` network (`pwff`).
3.  **Dropout**: Dropout is applied.
4.  **Residual Connection**: The output of the dropout layer is added back to the input of this block.
5.  **Final Normalization (Optional)**: If `norm_first` is false, a final normalization step is applied with `norm_2`.

### An ASCII Diagram of a `TransformerEncoderLayer`

```
          Input
            |
.-----------'-----------.
|                       |
| (Optional Norm)       |
|                       |
|  Multi-Head Attention |
|                       |
|       Dropout         |
|                       |
`-----------(+)---------'  <- Residual Connection 1
            |
            |
.-----------'-----------.
|                       |
|     Layer Norm        |
|                       |
| Position-Wise FFN     |
|                       |
|       Dropout         |
|                       |
`-----------(+)---------'  <- Residual Connection 2
            |
            |
    (Optional Norm)
            |
            V
          Output
```
### Code Example: Configuration and Masking

Building and using a `TransformerEncoder` involves two key steps: configuring it correctly and providing the necessary attention masks.

#### Configuration

You configure a `TransformerEncoder` using `TransformerEncoderConfig`. This allows you to set all the important hyperparameters.

```rust
use burn::prelude::*;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig};

fn create_transformer<B: Backend>(device: &B::Device) -> TransformerEncoder<B> {
    let config = TransformerEncoderConfig::new(
        /* d_model */ 256,
        /* d_ff */    1024,
        /* n_heads */ 8,
        /* n_layers */ 6,
    )
    .with_dropout(0.1)
    .with_norm_first(true); // Use Pre-LayerNorm

    config.init(device)
}
```

#### Masking

Masking is a critical concept in transformers. It tells the self-attention mechanism which tokens it should be allowed to "look at."

*   **Padding Mask (`mask_pad`)**: When you process a batch of sequences of different lengths, you pad the shorter sequences with a special "padding" token. The padding mask is a boolean tensor that tells the attention mechanism to ignore these padding tokens, so they don't influence the result.

*   **Attention Mask (`mask_attn`)**: This is a more general mask. For a generative LLM, you need an **autoregressive** or **causal** mask. This is a triangular mask that prevents a token at a given position from attending to any subsequent tokens. This is crucial because during inference, the model should only have access to the tokens it has already generated.

Burn provides a helper function to generate this mask:

```rust
use burn::nn::attention::generate_autoregressive_mask;
use burn::prelude::*;

fn create_causal_mask<B: Backend>(batch_size: usize, seq_length: usize, device: &B::Device) -> Tensor<B, 3, Bool> {
    generate_autoregressive_mask(batch_size, seq_length, device)
}
```

When you call the `forward` method, you pass these masks to the `TransformerEncoderInput`.

```rust
use burn::nn::transformer::TransformerEncoderInput;
use burn::prelude::*;

fn run_forward_pass<B: Backend>(
    transformer: &TransformerEncoder<B>,
    input_tensor: Tensor<B, 3>,
    pad_mask: Tensor<B, 2, Bool>, // Shape: [batch_size, seq_length]
    attn_mask: Tensor<B, 3, Bool>, // Shape: [batch_size, seq_length, seq_length]
) -> Tensor<B, 3> {
    let input = TransformerEncoderInput::new(input_tensor)
        .mask_pad(pad_mask)
        .mask_attn(attn_mask);

    transformer.forward(input)
}
```

This modular and composable design makes Burn's transformer implementation easy to understand, modify, and extend. By combining these powerful `Module` building blocks, you can construct sophisticated LLMs and other sequence-based models with confidence.

---

## Exercises

1.  **Explore `MultiHeadAttention`**:
    a.  Navigate to the `crates/burn-nn/src/modules/attention/mod.rs` file.
    b.  Look at the `MultiHeadAttention` struct. What are the main `Module` fields it contains? (Hint: You should see `Linear` layers for Queries, Keys, Values, and the output).
    c.  How is `d_model` and `n_heads` used to determine the dimensions of these linear layers?
2.  **Implement a Padding Mask**:
    a.  Write a function `create_padding_mask<B: Backend>(token_ids: &Tensor<B, 2, Int>, pad_token_id: i64) -> Tensor<B, 2, Bool>`.
    b.  This function should take a batch of token ID sequences (shape `[batch_size, seq_length]`) and return a boolean mask of the same shape where the value is `true` for every position that is *not* the `pad_token_id` and `false` where it is.
3.  **Thought Experiment**: The `TransformerEncoder` is suitable for models like BERT or GPT. However, for a sequence-to-sequence task (like translation), you would also need a `TransformerDecoder`. What is the key difference between a self-attention layer in an encoder and one in a decoder? (Hint: It involves another type of attention, often called "cross-attention").
