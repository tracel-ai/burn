//! Whether a kernel may write a buffer in place, decided the same way on every run of a graph
//! capture.

use cubecl::{client::Client, server::Handle};

/// Whether a kernel may write `handle`'s buffer in place: nothing but the memory pool and this
/// handle holds it.
///
/// The count [`Handle::can_mut`] reads also takes in the clones the device thread holds while it
/// works through what it was sent, such as the handles it clones as it sets up an allocation.
/// Those come and go with the device thread's progress, so the same buffer at the same point of
/// the same work can be refused in one run and written in place in the next. Eager work only
/// writes out of place once more for it. A graph capture cannot: the recorded run must allocate
/// what its planning runs did, and an operation that writes in place in one run and allocates in
/// the other breaks that. So while `client`'s stream captures, a refusal waits for the device
/// thread to finish everything sent to it so far, which lets those clones go, and asks again:
/// what still holds the buffer then is a holder of the work's own.
pub fn can_write_in_place(client: &Client, handle: &Handle) -> bool {
    if handle.can_mut() {
        return true;
    }
    if !client.is_capturing() {
        return false;
    }
    // A blocking round trip that checks no buffer: it returns once the device thread has run
    // every task sent before it, and is allowed while a capture records.
    let _ = client.check(core::iter::empty());
    handle.can_mut()
}
