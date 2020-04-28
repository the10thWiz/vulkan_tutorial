

// trait Shader<GE, Fss2> where GE: GraphicsEntryPointAbstract<SpecializationConstants = Fss2> {
//     fn main_entry_point(self);
// }
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::shader::ShaderModule;

mod vertex_shader {
    vulkano_shaders::shader! {
       ty: "vertex",
       path: "src/shader/uniformbuffer.vert"
    }
}

// mod fragment_shader {
//     vulkano_shaders::shader! {
//         ty: "fragment",
//         path: "src/shader/uniformbuffer.frag"
//     }
// }

// let vert_shader_module = vertex_shader::Shader::load(device.clone())
//     .expect("failed to create vertex shader module!");
// let frag_shader_module = fragment_shader::Shader::load(device.clone())
//     .expect("failed to create fragment shader module!");

pub struct Shader<S> {
    sh: S
}

// impl<S, E: GraphicsEntryPointAbstract> Shader<S> {
//     fn main_entry(&self) -> E {
        
//     }
// }

pub fn get_vertex_shader(device: &Arc<Device>) -> Arc<ShaderModule> {

    let vert_shader_module = vertex_shader::Shader::load(device.clone())
    .expect("failed to create vertex shader module!");
    vert_shader_module.module().clone()
}
