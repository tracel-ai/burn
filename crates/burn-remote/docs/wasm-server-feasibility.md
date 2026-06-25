# Browser-as-compute-peer: feasibility spike

Status: **investigation only — no implementation yet.** This records whether a browser can act as
a Burn Remote *server* (compute peer that accepts connections), which gates the "volunteer GPU pool"
and "browser-to-browser compute" ideas.

## The gating question

A browser compute peer must **accept inbound connections**, not just dial out (which the client
already does). Browsers have no UDP/QUIC, so iroh tunnels through a **relay over WebSocket**. The
question was: can an iroh wasm `Endpoint` *accept* connections forwarded by its relay, or is the
accept side compiled out / non-functional in the browser?

## Verdict: yes, iroh supports accept in the browser

Evidence from `iroh 1.0.0` (paths relative to the crates.io source):

- **`wasm_browser` activates automatically on our target.** It is a `cfg_aliases!` alias in
  `build.rs`: `wasm_browser: { all(target_family = "wasm", target_os = "unknown") }`. So
  `wasm32-unknown-unknown` — the target our wasm client already uses — enables the browser code
  paths with no extra `RUSTFLAGS`.
- **`Endpoint::accept()` is not gated out.** `src/endpoint.rs:1152` `pub fn accept(&self) -> Accept`
  has no `#[cfg(not(wasm_browser))]`; it delegates to the underlying QUIC endpoint's accept.
- **`Router` (the accept-loop helper) is not gated out.** `src/protocol.rs` defines `Router`,
  `RouterBuilder::accept`, and `spawn` with no `wasm_browser` gating.
- **The relay transport is duplex and present in wasm.** In `src/socket/transports.rs`, `mod ip;`
  (UDP) is `#[cfg(not(wasm_browser))]`, but `mod relay;` is unconditional, and the wasm transport
  set is `(custom, relay)`. `RelayTransport` (`src/socket/transports/relay.rs`) has both
  `poll_recv` (receives datagrams the relay forwards) and `RelaySender` — so a browser endpoint can
  *receive* inbound packets, hence inbound connections.
- **iroh spawns its accept tasks with `spawn_local` in wasm.** `src/runtime.rs:99`
  `wasm_bindgen_futures::spawn_local(future)` under `#[cfg(wasm_browser)]`. Crucially, the wasm
  spawn does **not** require `Send` futures, unlike the native multi-thread spawn.

Reachability detail: a browser peer is addressed by `EndpointId` + its home relay URL. To reach it,
a dialer connects to that relay (browsers can open a WebSocket to any relay URL) and the relay
forwards packets to the browser over its existing connection. No direct path or hole-punching is
required — relay-only connectivity is sufficient.

## So what actually blocks a browser compute peer?

**Not iroh.** The blocker is entirely on *our* side: `burn-remote`'s `serve` path is native-only.
Making it work in the browser is the same shape of port already done for the client:

1. **Task spawning.** The server's accept loop and per-session workers use native `tokio::spawn`
   (which requires `Send`) on a multi-thread runtime. In wasm they must use `spawn_local` (no `Send`
   bound — which single-threaded wasm makes *easier*, not harder, same as the client port).
2. **Drop `burn-communication`.** The websocket server (axum/tokio-net) cannot build for wasm; the
   wasm server path must use the iroh transport only, exactly like the wasm client.
3. **Compute backend = WebGPU.** `burn-wgpu`'s WebGPU target already runs in wasm, so the actual
   "GPU" half of a browser compute peer is solved.
4. **Feature gating.** Today enabling `server` on wasm fails on `mio`; a wasm-capable server would be
   a separate, relay-only server path gated to `wasm32` + `iroh`, mirroring the client.

## Recommended path

- **#7 (browser-to-browser, 1:1) first** — the minimal milestone: one browser serves over iroh,
  another peer (browser or native) connects. It proves the port end to end and depends on nothing
  else. Estimated as a bounded spike given the client port is the template.
- **#6 (volunteer pool)** is then #7 + a coordinator (liveness, work assignment, churn) + a
  distribution layer (#4 collectives or #5 graph replay) + the public-data-only privacy constraint.

## Porting plan

Mapping the `serve` path showed the work is larger than a single change, and the central obstacle
is the session worker, not Iroh. Staged so each step keeps native green:

1. **Task spawning (done).** A `server::spawn::spawn_detached` helper splits native `tokio::spawn`
   from browser `spawn_local`, mirroring the client. The Iroh session and transfer streams route
   through it, and the session response writer signals completion over a `oneshot` instead of a
   joined handle so both targets share one path.
2. **Session worker (done).** The worker's loop is now an `async fn run`; a target-split `drive`
   keeps native's dedicated thread + `block_on` and adds a browser `spawn_local` driver, and the
   inner readback spawns route through `spawn_detached`. A browser peer cannot host co-located
   collective / same-host-transfer participants (single event-loop thread), which is documented in
   the worker module; ordinary compute is unaffected.
3. **Feature split (done).** The `server` feature is now portable (`tokio/sync` only); native targets
   get `rt-multi-thread`, `time`, and `signal` from a `cfg(not(wasm))` dependency table, and
   `burn-communication` is pulled only by `websocket`. The Iroh path was decoupled from
   `burn-communication` (its shutdown signal is now a small `tokio::signal` helper), the blocking
   `start_iroh` / `start_iroh_async` entry points are native-only, and `transfer.rs`'s timers go
   through `server::time` (tokio on native, `gloo-timers` in the browser). The crate now builds and
   passes clippy for `wasm32-unknown-unknown` with `--features server,iroh,client`.
4. **Browser WebGPU example (done).** [`examples/remote-compute-peer-web`](../../../examples/remote-compute-peer-web)
   is a browser compute peer: it brings up a WebGPU device, binds an Iroh endpoint, and serves
   tensor ops, compiled to wasm. WebGPU needed a one-line `cubecl-runtime` fix (gating
   `ComputeClient::read_lazy` to non-wasm, matching its async twin), carried on the `jwric/cubecl`
   fork the workspace points at.

## Umbrella API (done)

`burn::server::serve(device, node)` is the portable, non-blocking server entry and works in the
browser. Getting there meant decoupling the server from the websocket transport across the feature
graph: `server` (in `burn-dispatch`, `burn-tensor`, `burn`) now pulls the Iroh server core only
(`remote-server = ["remote-iroh", ...]`), and the legacy websocket server — `Channel::WebSocket`,
`start`/`start_async`'s websocket arm, and the `start_websocket*` dispatch entries — is gated behind
the `remote` feature. The blocking `start`/`start_async` helpers are additionally native-only. The
browser example serves through `burn::server::serve` on a `Device::wgpu_async` WebGPU device.

## Not yet done

- A live browser-serves / native-connects end-to-end run (needs a browser + relay; verified here
  only by compilation).

## Honest constraints (unchanged by this finding)

- **Relays are always in the path** — "serverless" means no *application* backend, but iroh relays
  (public or self-hosted) still carry browser traffic.
- **WebGPU peers are weak servers** — memory-capped, no large models, and browsers throttle
  background tabs, so a backgrounded peer stalls. Pushes #6 toward small models or shards.
- **Privacy** — a compute peer sees plaintext data and ops, so #6 fits *public* compute (open-model
  training on public data), not private inputs.
