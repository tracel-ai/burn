use crate::context::WorkGroupSize;

#[derive(new, Debug, PartialEq, Eq, Hash)]
pub struct RenderOptions {
    pub(crate) workgroup_size: WorkGroupSize,
    pub(crate) type_name_float: String,
    pub(crate) type_name_int: String,
}

impl RenderOptions {
    pub fn id(&self) -> String {
        format!(
            "wgz-{}-{}-{}-tf-{}-ti-{}",
            self.workgroup_size.x,
            self.workgroup_size.y,
            self.workgroup_size.z,
            self.type_name_float,
            self.type_name_int
        )
    }
}

pub trait KernelTemplate {
    fn id(&self) -> String;
    fn render(&self) -> String;
}

#[macro_export]
macro_rules! kernel_wgsl {
    (
        $struct:ident,
        $file:expr
    ) => {
        #[derive(new)]
        pub struct $struct {
            options: RenderOptions,
        }

        impl KernelTemplate for $struct {
            fn id(&self) -> String {
                $file.to_string() + self.options.id().as_str()
            }

            fn render(&self) -> String {
                let source = include_str!($file);

                let size = &self.options.workgroup_size;
                let source = source.replace("WORKGROUP_SIZE_X", &size.x.to_string());
                let source = source.replace("WORKGROUP_SIZE_Y", &size.y.to_string());
                let source = source.replace("WORKGROUP_SIZE_Z", &size.z.to_string());

                let source = source.replace("float", &self.options.type_name_float);
                let source = source.replace("int", &self.options.type_name_int);

                source
            }
        }
    };
}

kernel_wgsl!(ElemwiseRaw, "./template/elemwise.wgsl");

#[macro_export]
macro_rules! kernel_elemwise {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct {
            raw: $crate::kernel::ElemwiseRaw,
        }

        impl $struct {
            pub fn new(options: $crate::kernel::RenderOptions) -> Self {
                Self {
                    raw: $crate::kernel::ElemwiseRaw::new(options),
                }
            }
        }
        impl KernelTemplate for $struct {
            fn id(&self) -> String {
                let id = self.raw.id();
                id + $ops
            }

            fn render(&self) -> String {
                let source = self.raw.render();
                source.replace("OPS", $ops)
            }
        }
    };
}
