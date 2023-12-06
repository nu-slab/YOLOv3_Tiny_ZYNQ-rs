use anyhow::Result;
use std::time::Instant;

use tiny_yolo_v3_zynq_rs::img_proc::{letterbox_img, draw_bbox};
use tiny_yolo_v3_zynq_rs::yolo::YoloV3Tiny;


fn main() -> Result<()> {
    let wdir = "examples/weights";
    let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wdir, wdir)?;

    let test_img = image::open("examples/t19.jpg")?;

    let start = Instant::now();

    let result = yolo.start(test_img, 0)?;

    let end = start.elapsed();
    let t = end.as_secs_f64() * 1000.0;
    println!("{:?}", result);
    println!("Processing time:{:.03}ms, {:.1}FPS", t, 1000. / t);

    let test_img = image::open("examples/t19.jpg")?;
    let mut lb_img = letterbox_img(test_img, 416, 0);

    draw_bbox(&mut lb_img, &result);
    lb_img.save("out.png")?;

    Ok(())
}
