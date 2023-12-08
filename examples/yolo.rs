use anyhow::Result;
use std::time::Instant;

use yolo_v3_tiny_zynq::img_proc::draw_bbox;
use yolo_v3_tiny_zynq::yolov3_tiny::YoloV3Tiny;

fn main() -> Result<()> {
    // 重みファイルがあるディレクトリ
    let wpath = "examples/weights.tar.gz";

    // YOLOのモデルを初期化
    let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wpath)?;

    // テスト画像を読み込む
    let test_img = image::open("examples/t19.jpg")?;

    let start = Instant::now();

    // YOLOの処理を開始
    let result = yolo.start(&test_img, 0)?;

    let end = start.elapsed();
    let t = end.as_secs_f64() * 1000.0;
    println!("{:?}", result);
    println!("Processing time:{:.03}ms, {:.1}FPS", t, 1000. / t);

    // BBox描画のためDynamicImageをRGB画像に変換
    let mut rgb_img = test_img.to_rgb8();
    draw_bbox(&mut rgb_img, &result, 20., 6.);

    // 画像を保存
    std::fs::create_dir_all("./out")?;
    rgb_img.save("./out/out.png")?;

    Ok(())
}
