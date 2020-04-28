use std::io::Cursor;
use std::sync::Arc;

use png;
use vulkano::command_buffer::{AutoCommandBuffer, CommandBufferExecFuture};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{Dimensions, ImmutableImage};
use vulkano::sync::NowFuture;

fn load_texture(
    queue: Arc<Queue>,
) -> (
    Arc<ImmutableImage<Format>>,
    CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
) {
    let png_bytes = include_bytes!("images/image_img.png").to_vec();
    let cursor = Cursor::new(png_bytes);
    let decoder = png::Decoder::new(cursor);
    let (info, mut reader) = decoder.read_info().unwrap();
    let dimensions = Dimensions::Dim2d {
        width: info.width,
        height: info.height,
    };
    let mut image_data = Vec::new();
    image_data.resize((info.width * info.height * 4) as usize, 0);
    reader.next_frame(&mut image_data).unwrap();

    ImmutableImage::from_iter(
        image_data.iter().cloned(),
        dimensions,
        Format::R8G8B8A8Srgb,
        queue.clone(),
    )
    .unwrap()
}
