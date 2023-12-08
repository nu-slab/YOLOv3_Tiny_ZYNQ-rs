use anyhow::Result;
use std::time::Instant;

use yolo_v3_tiny_zynq::img_proc::{draw_bbox, letterbox_img_with_patial_enlargement};
use yolo_v3_tiny_zynq::yolov3_tiny::YoloV3Tiny;

fn main() -> Result<()> {
    let wpath = "examples/weights.tar.gz";

    let rotate_angle = 90;

    let crop_x = None;
    let crop_y = Some(75);
    let crop_w = 75;
    let crop_h = 150;

    // YOLO IP を初期化
    let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wpath)?;

    // ./out ディレクトリを作成
    std::fs::create_dir_all("./out")?;
    // テスト画像を読み込む
    let img = image::open("examples/pe_sample.png")?;

    let start = Instant::now();

    // YOLOとプロットに画像を使い回すため，事前に回転させる
    let rotated = img.rotate90();

    // YOLOの処理を開始 (事前に回転しているため，rotate_enはfalse)
    let result = yolo.start_with_patial_enlargement(
        &rotated,
        rotate_angle,
        false,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
    )?;

    // 画像を変形してBBox描画 (事前に回転しているため，rotate_enはfalse)
    let mut rgb_img = letterbox_img_with_patial_enlargement(
        &rotated,
        rotate_angle,
        false,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
    );
    draw_bbox(&mut rgb_img, &result, 20., 4.);

    let end = start.elapsed();
    let t = end.as_secs_f64() * 1000.0;
    println!("Processing time:{:.03}ms, {:.1}FPS", t, 1000. / t);

    // 画像を保存
    rgb_img.save(format!("./out/out.png"))?;

    Ok(())
}
