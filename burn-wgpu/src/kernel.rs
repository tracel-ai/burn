use crate::context::WorkGroupSize;

#[derive(new, Debug, PartialEq, Eq, Hash)]
pub struct RenderOptions {
    pub(crate) workgroup_size: WorkGroupSize,
    pub(crate) type_name_elem: Option<String>,
    pub(crate) type_name_int: Option<String>,
}

impl RenderOptions {
    pub fn id(&self) -> String {
        format!(
            "wgz-{}-{}-{}-te-{:?}-ti-{:?}",
            self.workgroup_size.x,
            self.workgroup_size.y,
            self.workgroup_size.z,
            self.type_name_elem,
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
                let mut source = source.replace("WORKGROUP_SIZE_X", &size.x.to_string());
                source = source.replace("WORKGROUP_SIZE_Y", &size.y.to_string());
                source = source.replace("WORKGROUP_SIZE_Z", &size.z.to_string());

                if let Some(ty_name) = &self.options.type_name_elem {
                    source = source.replace("elem", ty_name);
                }

                if let Some(ty_name) = &self.options.type_name_int {
                    source = source.replace("int", ty_name);
                }

                source
            }
        }
    };
}
