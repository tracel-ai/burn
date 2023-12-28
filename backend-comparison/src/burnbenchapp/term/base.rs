use crate::burnbenchapp::Application;

#[derive(new)]
pub struct TermApplication;

impl Application for TermApplication {
    fn init(&mut self)  {
    }

    fn run(&mut self) {
        println!("Hello World !")
    }

    fn cleanup(&mut self) {
    }
}
