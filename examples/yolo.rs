use anyhow::Result;
use std::time::Instant;

use tiny_yolo_v3_zynq_rs::img_proc::draw_bbox;
use tiny_yolo_v3_zynq_rs::yolov3_tiny::YoloV3Tiny;

fn main() -> Result<()> {
    let wdir = "examples/weights";
    let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wdir, wdir)?;

    let test_img = image::open("examples/t19.jpg")?;

    let start = Instant::now();

    let result = yolo.start(&test_img, 0)?;

    let end = start.elapsed();
    let t = end.as_secs_f64() * 1000.0;
    println!("{:?}", result);
    println!("Processing time:{:.03}ms, {:.1}FPS", t, 1000. / t);

    let mut rgb_img = test_img.to_rgb8();
    draw_bbox(&mut rgb_img, &result, 20., 6.);

    std::fs::create_dir_all("./out")?;
    rgb_img.save("./out/out.png")?;

    Ok(())
}
