# Advanced LSTM Implementation with Burn
A sophisticated implementation of Long Short-Term Memory (LSTM) networks in Burn, featuring state-of-the-art architectural enhancements and optimizations. This implementation includes bidirectional processing capabilities and advanced regularization techniques, making it suitable for both research and production environments. More details can be found at the [PyTorch implementation](https://github.com/shiv08/Advanced-LSTM-Implementation-with-PyTorch).

`LstmNetwork` is the top-level module with bidirectional support and output projection. It can support multiple LSTM variants by setting appropriate `bidirectional` and `num_layers`：
* LSTM: `num_layers = 1` and `bidirectional = false`
* Stacked LSTM: `num_layers > 1` and `bidirectional = false`
* Bidirectional LSTM: `num_layers = 1` and `bidirectional = true`
* Bidirectional Stacked LSTM: `num_layers > 1` and `bidirectional = true`

This implementation is complementary to Burn's official LSTM, users can choose either one depends on the project's specific needs.

## Example Usage

### Training
```sh
cargo run --release --features ndarray -- train --artifact-dir /home/wangjw/data/work/projects/lstm/output --num-epochs 30 --batch-size 32 --num-workers 2 --lr 0.001
```

### Inference
```sh
cargo run --release --features ndarray -- infer --artifact-dir /home/wangjw/data/work/projects/lstm/output
```
