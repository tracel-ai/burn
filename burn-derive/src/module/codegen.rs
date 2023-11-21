use proc_macro2::TokenStream;

/// Basic trait to be implemented for Module generation.
pub(crate) trait ModuleCodegen {
    fn gen_num_params(&self) -> TokenStream;
    fn gen_visit(&self) -> TokenStream;
    fn gen_devices(&self) -> TokenStream;
    fn gen_to_device(&self) -> TokenStream;
    fn gen_fork(&self) -> TokenStream;
    fn gen_map(&self) -> TokenStream;
    fn gen_valid(&self) -> TokenStream;
    fn gen_into_record(&self) -> TokenStream;
    fn gen_load_record(&self) -> TokenStream;
    fn gen_clone(&self) -> TokenStream;
}
