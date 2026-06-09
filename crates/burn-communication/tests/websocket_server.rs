//! Integration tests for the websocket [`WsServer`] channel implementation.
//!
//! These drive a real `WsServer` (bound to an ephemeral `127.0.0.1:0` port read back from the
//! listener) with a real `tokio-tungstenite` client, exercising the concurrency and
//! disconnect behaviour of the channel. Every await that could hang is wrapped in a timeout so
//! a regression fails fast instead of blocking CI.
//!
//! Run with: `cargo test -p burn-communication --features websocket`.
#![cfg(feature = "websocket")]

use std::time::Duration;

use burn_communication::websocket::{WsServer, WsServerChannel};
use burn_communication::{CommunicationChannel, Message, ProtocolServer};
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

/// Generous cap on every awaited operation: a passing test finishes in milliseconds; a
/// regression (panic, deadlock, dropped connection) trips the timeout instead of hanging.
const TIMEOUT: Duration = Duration::from_secs(10);

type Client = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// How a single server-side `recv` resolved — used by handlers that report back to the test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Recv {
    Message,
    End,
    Error,
}

fn classify<E>(result: Result<Option<Message>, E>) -> Recv {
    match result {
        Ok(Some(_)) => Recv::Message,
        Ok(None) => Recv::End,
        Err(_) => Recv::Error,
    }
}

/// A running test server bound to an ephemeral port, with a shutdown the test controls.
struct TestServer {
    port: u16,
    shutdown: Option<oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    /// Bind an ephemeral port, let `build` register the routes, and serve on it. Binding
    /// happens before the serve task is spawned, so the port is in the listen backlog by the
    /// time this returns and clients can connect without a readiness race.
    async fn start(build: impl FnOnce(WsServer) -> WsServer) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let port = listener.local_addr().expect("local_addr").port();

        // The port passed to `new` is unused — we serve on our own pre-bound listener.
        let server = build(WsServer::new(0));

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let handle = tokio::spawn(async move {
            let shutdown = async move {
                let _ = shutdown_rx.await;
            };
            if let Err(err) = server.serve_on(listener, shutdown).await {
                eprintln!("test server error: {err:?}");
            }
        });

        Self {
            port,
            shutdown: Some(shutdown_tx),
            handle,
        }
    }

    fn url(&self, path: &str) -> String {
        format!(
            "ws://127.0.0.1:{}/{}",
            self.port,
            path.trim_start_matches('/')
        )
    }

    async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = timeout(TIMEOUT, self.handle).await;
    }
}

/// A route handler that echoes every binary message back, looping until the stream ends.
async fn echo(mut channel: WsServerChannel) {
    while let Ok(Some(msg)) = channel.recv().await {
        if channel.send(msg).await.is_err() {
            break;
        }
    }
}

// --- client helpers -----------------------------------------------------------------------

async fn connect(url: &str) -> Client {
    let (ws, _resp) = timeout(TIMEOUT, connect_async(url))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    ws
}

async fn send_binary(ws: &mut Client, data: &[u8]) {
    timeout(TIMEOUT, ws.send(WsMessage::Binary(data.to_vec().into())))
        .await
        .expect("send timed out")
        .expect("send failed");
}

/// Read until the next binary frame, skipping pongs/pings/text the server may interleave.
async fn recv_binary(ws: &mut Client) -> Vec<u8> {
    let fut = async {
        loop {
            match ws.next().await {
                Some(Ok(WsMessage::Binary(data))) => return data.to_vec(),
                Some(Ok(_)) => continue,
                Some(Err(err)) => panic!("client recv error: {err:?}"),
                None => panic!("client stream ended before a binary frame"),
            }
        }
    };
    timeout(TIMEOUT, fut).await.expect("recv_binary timed out")
}

/// Drive a clean websocket close handshake to completion from the client side.
async fn close_cleanly(mut ws: Client) {
    let _ = ws.close(None).await;
    // Drain so the server's close echo is read and the handshake completes on both sides.
    while let Some(Ok(_)) = ws.next().await {}
}

// --- tests --------------------------------------------------------------------------------

/// Many clients at once each get their own response — proves connections aren't serialized.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_connections_are_not_serialized() {
    let server = TestServer::start(|s| s.route("/echo", echo)).await;
    let url = server.url("echo");

    // Spawn all clients eagerly (collect forces the spawns) so they run concurrently, then
    // await them — rather than spawn-then-await one at a time, which would serialize them.
    let clients: Vec<_> = (0..50)
        .map(|i| {
            let url = url.clone();
            tokio::spawn(async move {
                let mut ws = connect(&url).await;
                let payload = format!("hello-{i}").into_bytes();
                send_binary(&mut ws, &payload).await;
                assert_eq!(
                    recv_binary(&mut ws).await,
                    payload,
                    "client {i} echo mismatch"
                );
                close_cleanly(ws).await;
            })
        })
        .collect();

    for (i, client) in clients.into_iter().enumerate() {
        timeout(TIMEOUT, client)
            .await
            .unwrap_or_else(|_| panic!("client {i} timed out"))
            .unwrap_or_else(|_| panic!("client {i} panicked"));
    }

    server.shutdown().await;
}

/// One peer vanishing abruptly (TCP drop, no close frame) must not take down the others.
/// Regression for bug #1: the old `None => todo!()` would panic the handler on disconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn abrupt_disconnect_does_not_affect_other_connections() {
    // Echo handler that also reports how its stream ended, so the test can assert the
    // disconnected peer's handler terminated rather than panicked or hung.
    let (report_tx, mut report_rx) = mpsc::unbounded_channel::<Recv>();
    let server = TestServer::start(move |s| {
        s.route("/echo", move |mut channel: WsServerChannel| {
            let report = report_tx.clone();
            async move {
                loop {
                    match channel.recv().await {
                        Ok(Some(msg)) => {
                            if channel.send(msg).await.is_err() {
                                let _ = report.send(Recv::Error);
                                return;
                            }
                        }
                        Ok(None) => {
                            let _ = report.send(Recv::End);
                            return;
                        }
                        Err(_) => {
                            let _ = report.send(Recv::Error);
                            return;
                        }
                    }
                }
            }
        })
    })
    .await;
    let url = server.url("echo");

    let mut a = connect(&url).await;
    let mut b = connect(&url).await;

    // Both work to start with.
    send_binary(&mut a, b"a1").await;
    assert_eq!(recv_binary(&mut a).await, b"a1");
    send_binary(&mut b, b"b1").await;
    assert_eq!(recv_binary(&mut b).await, b"b1");

    // Abruptly drop A's TCP connection — no websocket close handshake.
    drop(a);

    // A's handler must terminate (without panicking); with the `todo!()` bug it would panic on
    // the bare-None path and never report, tripping this timeout.
    let ended = timeout(TIMEOUT, report_rx.recv())
        .await
        .expect("A's handler did not terminate — it likely panicked")
        .expect("report channel closed");
    assert!(
        matches!(ended, Recv::End | Recv::Error),
        "A's handler ended in an unexpected state: {ended:?}"
    );

    // B must still be alive and responsive after A vanished.
    send_binary(&mut b, b"b2").await;
    assert_eq!(
        recv_binary(&mut b).await,
        b"b2",
        "B died after A disconnected"
    );

    close_cleanly(b).await;
    server.shutdown().await;
}

/// Hit the bare end-of-stream arm directly. Regression for bug #1: a second `recv` after the
/// stream has terminated yields `None`, which the old `todo!()` turned into a panic.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stream_end_after_close_does_not_panic() {
    let (report_tx, mut report_rx) = mpsc::unbounded_channel::<(Recv, Recv)>();
    let server = TestServer::start(move |s| {
        s.route("/drain", move |mut channel: WsServerChannel| {
            let report = report_tx.clone();
            async move {
                // First recv sees the client's Close -> Ok(None). The stream is then
                // terminated, so the second recv hits the bare-None arm -> Ok(None) with the
                // fix, panic with the old `todo!()`.
                let first = classify(channel.recv().await);
                let second = classify(channel.recv().await);
                let _ = report.send((first, second));
            }
        })
    })
    .await;

    let ws = connect(&server.url("drain")).await;
    close_cleanly(ws).await;

    let (first, second) = timeout(TIMEOUT, report_rx.recv())
        .await
        .expect("/drain handler did not finish — it likely panicked on the bare-None path")
        .expect("report channel closed");
    assert_eq!(
        first,
        Recv::End,
        "close frame should make the first recv return Ok(None)"
    );
    assert_eq!(
        second,
        Recv::End,
        "a terminated stream should make recv return Ok(None), not panic"
    );

    server.shutdown().await;
}

/// A keepalive ping must be skipped, not treated as a fatal error. Regression for bug #2:
/// the old catch-all turned Ping/Pong/Text into `Err(UnknownMessage)`, killing the connection.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn keepalive_ping_does_not_kill_connection() {
    let server = TestServer::start(|s| s.route("/echo", echo)).await;
    let mut ws = connect(&server.url("echo")).await;

    // Send a keepalive ping, then a real message; the echo must still come back.
    timeout(
        TIMEOUT,
        ws.send(WsMessage::Ping(b"keepalive".to_vec().into())),
    )
    .await
    .expect("ping send timed out")
    .expect("ping send failed");

    send_binary(&mut ws, b"after-ping").await;
    assert_eq!(
        recv_binary(&mut ws).await,
        b"after-ping",
        "connection died after a keepalive ping"
    );

    close_cleanly(ws).await;
    server.shutdown().await;
}

/// A client-initiated close makes the server's `recv` return `Ok(None)` and the handler end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn normal_close_ends_recv_cleanly() {
    let (report_tx, mut report_rx) = mpsc::unbounded_channel::<Recv>();
    let server = TestServer::start(move |s| {
        s.route("/echo", move |mut channel: WsServerChannel| {
            let report = report_tx.clone();
            async move {
                loop {
                    match channel.recv().await {
                        Ok(Some(msg)) => {
                            if channel.send(msg).await.is_err() {
                                let _ = report.send(Recv::Error);
                                return;
                            }
                        }
                        Ok(None) => {
                            let _ = report.send(Recv::End);
                            return;
                        }
                        Err(_) => {
                            let _ = report.send(Recv::Error);
                            return;
                        }
                    }
                }
            }
        })
    })
    .await;

    let mut ws = connect(&server.url("echo")).await;
    send_binary(&mut ws, b"ping").await;
    assert_eq!(recv_binary(&mut ws).await, b"ping");
    close_cleanly(ws).await;

    let ended = timeout(TIMEOUT, report_rx.recv())
        .await
        .expect("handler did not finish after close")
        .expect("report channel closed");
    assert_eq!(
        ended,
        Recv::End,
        "a client Close should make recv return Ok(None)"
    );

    server.shutdown().await;
}
