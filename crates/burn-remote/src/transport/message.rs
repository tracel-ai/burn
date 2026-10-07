//! Messages carried in frames of at most [`MAX_FRAME_SIZE`] bytes.
//!
//! Past a session's handshake, a small message travels as one frame behind a tag byte. A larger
//! one opens with a frame giving its length, then follows as the segments it was encoded into, so
//! it is never copied to be sent and is read into one buffer allocated for it. The handshake itself
//! stays in whole frames: a peer on another protocol version reads its refusal from them.

use bytes::{Buf, BufMut, Bytes, BytesMut};

use super::link::{FrameSink, FrameSource, MAX_FRAME_SIZE};
use crate::shared::Encoded;

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
        if message.len() <= MessageHead::MAX_WHOLE {
            let whole = MessageHead::Whole(message.into_bytes());
            return self.frames.send(whole.into()).await;
        }
        let head = MessageHead::Sliced {
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
            MessageHead::Sliced { len } => self.recv_sliced(len).await.map(Some),
        }
    }

    async fn recv_sliced(&mut self, len: u64) -> Result<Bytes, String> {
        let len = usize::try_from(len)
            .map_err(|_| format!("Peer sent a message of {len} bytes, too large to address"))?;
        let mut message = vec![0; len];
        let mut filled = 0;
        while filled < len {
            let end = len.min(filled + MAX_FRAME_SIZE);
            match self.frames.recv_into(&mut message[filled..end]).await? {
                0 => return Err("Peer sent an empty frame in the middle of a message".into()),
                read => filled += read,
            }
        }
        Ok(message.into())
    }
}

/// The frame a message opens with.
#[derive(Debug)]
enum MessageHead {
    /// The whole message, behind its tag.
    Whole(Bytes),
    /// The length of a message whose bytes follow in frames of their own.
    Sliced { len: u64 },
}

impl MessageHead {
    /// The largest message sent whole: a larger one is never copied behind a tag.
    const MAX_WHOLE: usize = 64 * 1024;
    const MAX_SIZE: usize = 1 + Self::MAX_WHOLE;
    const WHOLE: u8 = 0;
    const SLICED: u8 = 1;
}

impl TryFrom<Bytes> for MessageHead {
    type Error = String;

    fn try_from(mut frame: Bytes) -> Result<Self, String> {
        if frame.is_empty() {
            return Err("Peer sent a message with no head".into());
        }
        match frame.get_u8() {
            Self::WHOLE => Ok(Self::Whole(frame)),
            Self::SLICED if frame.len() == size_of::<u64>() => Ok(Self::Sliced {
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
            MessageHead::Sliced { len } => {
                let mut frame = BytesMut::with_capacity(1 + size_of::<u64>());
                frame.put_u8(MessageHead::SLICED);
                frame.put_u64_le(len);
                frame.freeze()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[tokio::test]
    async fn a_small_message_is_one_frame() {
        let message = message_of(MessageHead::MAX_WHOLE);

        let frames = frames_of([message.clone()]).await;

        assert_eq!(frames.len(), 1);
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

    #[tokio::test]
    async fn a_peer_that_closes_in_the_middle_of_a_message_is_an_error() {
        let mut frames = frames_of([message_of(2 * MAX_FRAME_SIZE)]).await;
        frames.pop();
        let mut source = MessageSource::new(ScriptedFrames(frames.into()));

        let result = source.recv().await;

        assert!(result.is_err(), "{result:?}");
    }

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
