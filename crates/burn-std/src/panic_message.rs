use alloc::string::String;
use core::any::Any;

/// The message of a caught panic, read from its payload.
pub trait PanicMessage {
    /// The message `panic!` was given, or a placeholder when the payload is not a string.
    fn message(&self) -> &str;
}

impl PanicMessage for dyn Any + Send {
    // `panic!` with a literal carries a `&'static str`, and with format arguments a `String`.
    fn message(&self) -> &str {
        self.downcast_ref::<&'static str>()
            .copied()
            .or_else(|| self.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic payload>")
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn a_panic_gives_its_message_whether_literal_or_formatted() {
        let literal = std::panic::catch_unwind(|| panic!("out of memory")).unwrap_err();
        let formatted = std::panic::catch_unwind(|| panic!("device {} missing", 3)).unwrap_err();
        let not_a_string = std::panic::catch_unwind(|| std::panic::panic_any(3u32)).unwrap_err();

        assert_eq!(literal.message(), "out of memory");
        assert_eq!(formatted.message(), "device 3 missing");
        assert_eq!(not_a_string.message(), "<non-string panic payload>");
    }
}
