use crate::context::WorkGroupSize;

#[derive(new, Debug, PartialEq, Eq, Hash)]
pub struct RenderOptions {
    workgroup_size: WorkGroupSize,
    type_name_float: String,
    type_name_int: String,
}

pub trait KernelTemplate {
    fn id(&self) -> &str;
    fn render(&self, options: &RenderOptions) -> String;
}

#[macro_export]
macro_rules! kernel {
    (
        $struct:ident,
        $file:expr
    ) => {
        pub struct $struct;

        impl KernelTemplate for $struct {
            fn id(&self) -> &str {
                $file
            }

            fn render(&self, options: &RenderOptions) -> String {
                let source = include_str!($file);

                let size = &options.workgroup_size;
                let source = source.replace("WORKGROUP_SIZE_X", &size.x.to_string());
                let source = source.replace("WORKGROUP_SIZE_Y", &size.y.to_string());
                let source = source.replace("WORKGROUP_SIZE_Z", &size.z.to_string());

                let source = source.replace("FLOAT", &options.type_name_float);
                let source = source.replace("INT", &options.type_name_int);

                source
            }
        }
    };
}

kernel!(Add, "./template/add_scalar.wgsl");
