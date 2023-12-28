use super::App;

/// Base trait to define an application
pub(crate) trait Application {
    fn init(&mut self) {
    }

    fn run(&mut self) {
    }

    fn cleanup(&mut self) {
    }
}

pub fn run() {
    let mut app = App::new();
    app.init();
    app.run();
    app.cleanup();
}
