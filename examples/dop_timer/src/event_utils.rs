/// Simply instrumented event; to verify event reporting.
#[tracing::instrument]
pub(crate) fn example_instrumented_event() {
    tracing::info!("test event");

    let span = tracing::info_span!("test_span");
    let _guard = span.enter();
    tracing::info!("inside span");
}
