//! Messages carried in frames of at most [`MAX_FRAME_SIZE`] bytes.
//!
//! Past the handshake, a small message travels as one frame behind a tag byte. A larger one opens
//! with a frame giving its length, then follows as the segments it was encoded into, so it is never
//! copied to be sent; a message of several segments is read into one buffer from [`BUFFERS`]. The
//! handshake itself stays in bare frames: a peer on another protocol version reads its refusal from
//! them.

use bytes::{Buf, BufMut, Bytes, BytesMut};

use super::link::{FrameSink, FrameSource, MAX_FRAME_SIZE, MAX_WHOLE_MESSAGE_SIZE};
use crate::shared::{BUFFERS, Encoded};

/// Sends each message in as many frames as it takes.
pub struct MessageSink<S> {
    frames: S,
}

/// Receives the messages a [`MessageSink`] sends.
pub struct MessageSource<S> {
    frames: S,
}

impl<S: FrameSink> MessageSink<S> {
    pub fn new(frames: S) -> Self {
        Self { frames }
    }

    pub async fn send(&mut self, message: Encoded) -> Result<(), String> {
        if message.len() <= MAX_WHOLE_MESSAGE_SIZE {
            let whole = MessageHead::Whole(message.into_bytes());
            return self.frames.send(whole.into()).await;
        }
        let head = MessageHead::Segmented {
            len: message.len() as u64,
        };
        self.frames.send(head.into()).await?;
        for segment in message.into_segments() {
            self.frames.send(segment).await?;
        }
        Ok(())
    }

    /// Finish the stream; no more messages will be sent.
    pub async fn close(&mut self) -> Result<(), String> {
        self.frames.close().await
    }
}

impl<S: FrameSource> MessageSource<S> {
    pub fn new(frames: S) -> Self {
        Self { frames }
    }

    /// The next message, or `None` when the peer closes the stream between two messages.
    pub async fn recv(&mut self) -> Result<Option<Bytes>, String> {
        let Some(frame) = self.frames.recv(MessageHead::MAX_SIZE).await? else {
            return Ok(None);
        };
        match MessageHead::try_from(frame)? {
            MessageHead::Whole(message) => Ok(Some(message)),
            MessageHead::Segmented { len } => self.recv_segmented(len).await.map(Some),
        }
    }

    async fn recv_segmented(&mut self, len: u64) -> Result<Bytes, String> {
        let len = usize::try_from(len)
            .map_err(|_| format!("Peer sent a message of {len} bytes, too large to address"))?;
        if len <= MAX_FRAME_SIZE {
            self.recv_one_segment(len).await
        } else {
            self.recv_many_segments(len).await
        }
    }

    /// A message of one segment, handed on as the frame the transport delivered.
    async fn recv_one_segment(&mut self, len: usize) -> Result<Bytes, String> {
        match self.frames.recv(len).await? {
            Some(segment) if segment.len() == len => Ok(segment),
            Some(segment) => Err(format!(
                "Peer sent a message of {len} bytes in a segment of {}",
                segment.len()
            )),
            None => Err("Peer closed the stream in the middle of a message".into()),
        }
    }

    /// A message of several segments, read into one pooled buffer.
    async fn recv_many_segments(&mut self, len: usize) -> Result<Bytes, String> {
        let mut message = BUFFERS.take_to_overwrite(len).map_err(|_| {
            format!("Peer sent a message of {len} bytes, more than can be allocated")
        })?;
        let mut filled = 0;
        while filled < len {
            let end = len.min(filled + MAX_FRAME_SIZE);
            // Zeroed a frame at a time, so the read overwrites it while it is still in cache.
            if message.len() < end {
                message.resize(end, 0);
            }
            match self.frames.recv_into(&mut message[filled..end]).await? {
                0 => return Err("Peer sent an empty frame in the middle of a message".into()),
                read => filled += read,
            }
        }
        message.truncate(len);
        Ok(message.into_bytes())
    }
}

/// The frame a message opens with.
#[derive(Debug)]
enum MessageHead {
    /// The whole message, behind its tag.
    Whole(Bytes),
    /// The length of a message whose bytes follow in frames of their own; one that fits in a frame
    /// follows in exactly one.
    Segmented { len: u64 },
}

impl MessageHead {
    const MAX_SIZE: usize = 1 + MAX_WHOLE_MESSAGE_SIZE;
    const WHOLE: u8 = 0;
    const SEGMENTED: u8 = 1;
}

impl TryFrom<Bytes> for MessageHead {
    type Error = String;

    fn try_from(mut frame: Bytes) -> Result<Self, String> {
        if frame.is_empty() {
            return Err("Peer sent a message with no head".into());
        }
        match frame.get_u8() {
            Self::WHOLE => Ok(Self::Whole(frame)),
            Self::SEGMENTED if frame.len() == size_of::<u64>() => Ok(Self::Segmented {
                len: frame.get_u64_le(),
            }),
            tag => Err(format!(
                "Peer sent a message head with tag {tag} and {} bytes",
                frame.len()
            )),
        }
    }
}

impl From<MessageHead> for Bytes {
    fn from(head: MessageHead) -> Self {
        match head {
            MessageHead::Whole(message) => {
                let mut frame = BytesMut::with_capacity(1 + message.len());
                frame.put_u8(MessageHead::WHOLE);
                frame.extend_from_slice(&message);
                frame.freeze()
            }
            MessageHead::Segmented { len } => {
                let mut frame = BytesMut::with_capacity(1 + size_of::<u64>());
                frame.put_u8(MessageHead::SEGMENTED);
                frame.put_u64_le(len);
                frame.freeze()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::Encode;
    use serde::Serialize;
    use std::collections::VecDeque;

    #[tokio::test]
    async fn a_small_message_is_one_frame() {
        let message = message_of(MAX_WHOLE_MESSAGE_SIZE);

        let frames = frames_of([message.clone()]).await;

        assert_eq!(frames.len(), 1);
        assert_eq!(received(frames).await, [message.into_bytes()]);
    }

    #[tokio::test]
    async fn a_message_one_byte_over_the_whole_size_follows_its_head() {
        let message = message_of(MAX_WHOLE_MESSAGE_SIZE + 1);

        let frames = frames_of([message.clone()]).await;

        assert_eq!(frames.len(), 2);
        assert_eq!(received(frames).await, [message.into_bytes()]);
    }

    /// Its frame is the segment it was encoded into: sending copies none of its bytes.
    #[tokio::test]
    async fn a_message_of_one_frame_size_follows_its_head_uncopied() {
        let message = message_of(MAX_FRAME_SIZE);
        let segment = message.clone().into_bytes();

        let frames = frames_of([message]).await;

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].as_ptr(), segment.as_ptr());
        assert_eq!(received(frames).await, [segment]);
    }

    #[tokio::test]
    async fn a_message_one_byte_over_the_frame_size_takes_two_frames_after_its_head() {
        let message = message_of(MAX_FRAME_SIZE + 1);

        let frames = frames_of([message.clone()]).await;

        assert_eq!(frames.len(), 3);
        assert_eq!(received(frames).await, [message.into_bytes()]);
    }

    /// A small message after a large one, so a message that ran into the next would show.
    #[tokio::test]
    async fn messages_of_many_frames_arrive_whole_and_in_order() {
        let large = message_of(5 * MAX_FRAME_SIZE + 3);
        let small = Encoded::from(&b"after"[..]);

        let frames = frames_of([large.clone(), small.clone()]).await;

        assert_eq!(frames.len(), 1 + 6 + 1);
        assert_eq!(
            received(frames).await,
            [large.into_bytes(), small.into_bytes()]
        );
    }

    /// Encoded a field at a time, as task batches are, so its first segment fills before it ends.
    #[tokio::test]
    async fn a_message_encoded_in_small_pieces_arrives_whole() {
        let values = Values((0..30_000).map(|i| u64::MAX - i).collect());
        let expected = rmp_serde::to_vec(&values).unwrap();

        let frames = frames_of([values.encode().unwrap()]).await;

        assert_eq!(frames.len(), 2);
        assert_eq!(received(frames).await, [expected]);
    }

    #[tokio::test]
    async fn a_peer_that_closes_in_the_middle_of_a_message_is_an_error() {
        let mut frames = frames_of([message_of(2 * MAX_FRAME_SIZE)]).await;
        frames.pop();
        let mut source = MessageSource::new(ScriptedFrames(frames.into()));

        let result = source.recv().await;

        assert!(result.is_err(), "{result:?}");
    }

    #[tokio::test]
    async fn a_message_of_one_segment_cut_short_is_an_error() {
        let len = MAX_WHOLE_MESSAGE_SIZE + 1;
        let head = MessageHead::Segmented { len: len as u64 };
        let short = Bytes::from(vec![7; len - 1]);
        let mut source = MessageSource::new(ScriptedFrames([head.into(), short].into()));

        let result = source.recv().await;

        assert!(result.is_err(), "{result:?}");
    }

    #[tokio::test]
    async fn a_message_longer_than_can_be_allocated_is_an_error() {
        let head = MessageHead::Segmented { len: u64::MAX };
        let mut source = MessageSource::new(ScriptedFrames([head.into()].into()));

        let result = source.recv().await;

        assert!(result.is_err(), "{result:?}");
    }

    #[derive(Serialize)]
    struct Values(Vec<u64>);

    impl Encode for Values {}

    fn message_of(len: usize) -> Encoded {
        Encoded::from(&(0..len).map(|i| i as u8).collect::<Vec<_>>()[..])
    }

    async fn frames_of(messages: impl IntoIterator<Item = Encoded>) -> Vec<Bytes> {
        let mut sink = MessageSink::new(RecordedFrames::default());
        for message in messages {
            sink.send(message).await.unwrap();
        }
        sink.frames.0
    }

    async fn received(frames: Vec<Bytes>) -> Vec<Bytes> {
        let mut source = MessageSource::new(ScriptedFrames(frames.into()));
        let mut messages = Vec::new();
        while let Some(message) = source.recv().await.unwrap() {
            messages.push(message);
        }
        messages
    }

    #[derive(Default)]
    struct RecordedFrames(Vec<Bytes>);

    impl FrameSink for RecordedFrames {
        async fn send(&mut self, frame: Bytes) -> Result<(), String> {
            self.0.push(frame);
            Ok(())
        }

        async fn close(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    /// Hands out its frames, then ends the stream.
    struct ScriptedFrames(VecDeque<Bytes>);

    impl FrameSource for ScriptedFrames {
        async fn recv(&mut self, max_len: usize) -> Result<Option<Bytes>, String> {
            match self.0.pop_front() {
                Some(frame) if frame.len() > max_len => {
                    Err(format!("a frame of {} bytes, over {max_len}", frame.len()))
                }
                frame => Ok(frame),
            }
        }
    }
}
