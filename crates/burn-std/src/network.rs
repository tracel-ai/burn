//! # Common Network Utilities

/// Network download utilities.
pub mod downloader {
    use indicatif::{ProgressBar, ProgressState, ProgressStyle};
    use reqwest::Client;

    /// Download the file at the specified url.
    /// File download progress is reported with the help of a [progress bar](indicatif).
    ///
    /// # Arguments
    ///
    /// * `url` - The file URL to download.
    /// * `message` - The message to display on the progress bar during download.
    ///
    /// # Returns
    ///
    /// A vector of bytes containing the downloaded file data.
    ///
    /// A connection that drops mid-body is retried. Bytes already received are kept and the
    /// next attempt asks for the remainder with a `Range` header.
    #[tokio::main(flavor = "current_thread")]
    pub async fn download_file_as_bytes(url: &str, message: &str) -> Vec<u8> {
        const ATTEMPTS: u32 = 5;
        let client = Client::new();
        let msg = message.to_owned();
        let mut bytes: Vec<u8> = Vec::new();
        let mut total_size = 0u64;
        let mut progress: Option<ProgressBar> = None;
        let mut last_error = None;

        for _attempt in 1..=ATTEMPTS {
            let mut request = client.get(url);
            if !bytes.is_empty() {
                request = request.header(reqwest::header::RANGE, format!("bytes={}-", bytes.len()));
            }

            let response = match request.send().await {
                Ok(response) => response,
                Err(err) => {
                    last_error = Some(err);
                    continue;
                }
            };

            // A 200 ignores Range and restarts the body. A 206 continues it.
            if !bytes.is_empty() && response.status() == reqwest::StatusCode::OK {
                bytes.clear();
                if let Some(bar) = progress.take() {
                    bar.finish_and_clear();
                }
            }

            if bytes.is_empty() {
                total_size = response.content_length().unwrap();
                bytes.reserve(total_size as usize);
                let bar = ProgressBar::new(total_size);
                bar.set_style(
                    ProgressStyle::with_template(
                        "{msg}\n    {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({eta})",
                    )
                    .unwrap()
                    .with_key(
                        "eta",
                        |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                            write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                        },
                    )
                    .progress_chars("▬  "),
                );
                bar.set_message(msg.clone());
                progress = Some(bar);
            }

            let mut response = response;
            loop {
                match response.chunk().await {
                    Ok(Some(chunk)) => {
                        bytes.extend_from_slice(&chunk);
                        if let Some(bar) = progress.as_ref() {
                            bar.set_position(std::cmp::min(bytes.len() as u64, total_size));
                        }
                    }
                    Ok(None) if bytes.len() as u64 >= total_size => {
                        if let Some(bar) = progress.as_ref() {
                            bar.finish_with_message(msg.clone());
                        }
                        return bytes;
                    }
                    Ok(None) => break,
                    Err(err) => {
                        last_error = Some(err);
                        break;
                    }
                }
            }
        }

        panic!("download of {url} failed after {ATTEMPTS} attempts: {last_error:?}");
    }

    #[cfg(test)]
    mod tests {
        use super::download_file_as_bytes;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::thread;
        use std::time::{Duration, Instant};

        fn accept_within(listener: &TcpListener, timeout: Duration) -> TcpStream {
            let start = Instant::now();
            loop {
                match listener.accept() {
                    Ok((stream, _)) => return stream,
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        if start.elapsed() > timeout {
                            panic!("timed out waiting for a download connection");
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(err) => panic!("accept failed: {err}"),
                }
            }
        }

        fn read_headers(stream: &mut TcpStream) -> String {
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                buf.push(byte[0]);
                if buf.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            String::from_utf8_lossy(&buf).into_owned()
        }

        /// A peer that closes after the first bytes used to panic in `chunk().unwrap()`.
        /// The downloader must resume from the bytes already received.
        #[test]
        fn resumes_after_dropped_connection() {
            let body = b"hello world";
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let port = listener.local_addr().unwrap().port();

            thread::spawn(move || {
                let timeout = Duration::from_secs(5);
                let mut first = accept_within(&listener, timeout);
                first.set_read_timeout(Some(timeout)).expect("read timeout");
                let _ = read_headers(&mut first);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                first.write_all(head.as_bytes()).unwrap();
                first.write_all(&body[..5]).unwrap();
                first.flush().unwrap();
                drop(first);

                let mut second = accept_within(&listener, timeout);
                second
                    .set_read_timeout(Some(timeout))
                    .expect("read timeout");
                let headers = read_headers(&mut second);
                let start = headers
                    .lines()
                    .find(|line| line.to_ascii_lowercase().starts_with("range:"))
                    .and_then(|line| line.split('=').nth(1))
                    .and_then(|value| value.trim().trim_end_matches('-').parse::<usize>().ok())
                    .unwrap_or(0);
                let payload = body[start..].to_vec();
                let (status, extra) = if start == 0 {
                    ("200 OK", String::new())
                } else {
                    (
                        "206 Partial Content",
                        format!(
                            "Content-Range: bytes {start}-{}/{}\r\n",
                            body.len() - 1,
                            body.len()
                        ),
                    )
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                    payload.len()
                );
                second.write_all(head.as_bytes()).unwrap();
                second.write_all(&payload).unwrap();
                second.flush().unwrap();
            });

            let got =
                download_file_as_bytes(&format!("http://127.0.0.1:{port}/model.bin"), "model");
            assert_eq!(got, body);
        }
    }
}
