# Remote MNIST Inference on the Web

This example runs a Burn model **in the browser** while executing every tensor operation on a
**remote compute peer** reached over [Iroh](https://iroh.computer/). The browser holds only the
model definition and weights; convolutions, matrix multiplies and everything else run on the
peer's backend (CPU or GPU). Only the 28x28 input and the 10 output probabilities cross the wire.

It mirrors the [`mnist-inference-web`](../mnist-inference-web) demo, but swaps the local WebAssembly
backend for the `burn-remote` Iroh client.

## Why this is interesting

- **Web scripting and experimentation against real GPUs.** A browser tab can drive a CUDA/Metal/Vulkan
  backend without shipping a native build to the user.
- **Models larger than the browser sandbox.** Memory and compute live on the peer, so the browser
  is not bound by WebGPU limits or tab memory.
- **One model, many backends.** The same client code targets whatever backend the peer hosts.

## How it works

The browser and the compute peer find each other through a shared **topic** string. Both sides hash
the topic to derive the peer's Iroh identity, so no node id has to be copied around:

```
topic --blake3--> secret key --> peer endpoint identity
   (peer binds the secret key; the browser derives its public half)
```

Burn then binds an Iroh endpoint in the browser, opens an authenticated QUIC session to the peer
through a relay, and ships operations as they are submitted.

## Running it

### 1. Start a compute peer (native)

The `p2p-remote-training` server derives its identity from the topic the same way, and computes on
the CPU:

```sh
cargo run -p p2p-remote-training --example p2p-remote-training --release -- server burn-web
```

The last argument (`burn-web`) is the topic; it must match what you type in the browser.

### 2. Build the web client

```sh
cd examples/remote-inference-web
./build-for-web.sh
```

### 3. Serve and open

```sh
./run-server.sh
```

Open <http://localhost:8000>, enter the same topic, click **Connect**, then draw a digit. The
probabilities are computed on the peer and streamed back.

## Notes

- The compute peer is model-agnostic; it just executes operations. Swapping the model on the
  client side requires no change to the peer.
- `model.bpk` is the trained MNIST model from the [`mnist`](../mnist) example, identical to the one
  used by `mnist-inference-web`.
- Connecting through public relays requires outbound network access from both the browser and the
  peer. A browser reaches Iroh peers only through a relay, so a fully local setup runs its own
  relay and gives both sides `IrohRelays::Private`.
