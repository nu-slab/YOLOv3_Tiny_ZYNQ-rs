//! YOLOに関する画像処理モジュール

use image::imageops::FilterType;
use image::{DynamicImage, Pixel, Rgb, RgbImage};

use crate::detection_result::DetectionData;

use imageproc::drawing::draw_filled_rect_mut;
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

/// 画像を指定した角度で回転させます。
///
/// # Args
///
/// * `img` - 回転させる画像
/// * `angle` - 回転させる角度（90, 180, 270のみ対応）
///
/// # Return
///
/// * 回転させた画像
pub fn rotate_img(img: &DynamicImage, angle: u32) -> DynamicImage {
    match angle {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img.clone(),
    }
}

/// 画像のピクセルデータをベクタの指定した位置に配置します。
///
/// # Args
///
/// * `data` - 配置先のデータ (in-place)
/// * `img` - 配置する画像
/// * `size` - 配置先のデータのサイズ
/// * `x_offset` - x軸方向のオフセット
/// * `y_offset` - y軸方向のオフセット
pub fn place_pixels(
    data: &mut Vec<i16>,
    img: &DynamicImage,
    size: u32,
    x_offset: u32,
    y_offset: u32,
) {
    for (x, y, pixel) in img.to_rgb8().enumerate_pixels() {
        let base_addr = 4 * (x + x_offset + (y + y_offset) * size) as usize;
        data[base_addr] = i16::from(pixel[0]);
        data[base_addr + 1] = i16::from(pixel[1]);
        data[base_addr + 2] = i16::from(pixel[2]);
        // data[base_addr + 3] = 0x7FFF; // for Debug
    }
}

/// 画像をリサイズ・回転し、正方形に整形したYOLO入力データを生成します。
///
/// # Args
///
/// * `img` - リサイズと回転を行う画像
/// * `size` - リサイズ後の画像のサイズ
/// * `rotate_angle` - 回転させる角度
///
/// # Return
///
/// * リサイズ、回転、パディングを行った画像のピクセルデータ
pub fn letterbox(img: &DynamicImage, size: u32, rotate_angle: u32) -> Vec<i16> {
    let resized = img.resize(size, size, FilterType::Nearest);
    let rotated = rotate_img(&resized, rotate_angle);

    let pad_w = rotated.width().abs_diff(size) / 2;
    let pad_h = rotated.height().abs_diff(size) / 2;

    let mut new_img = vec![0; (size * size * 4) as usize];
    place_pixels(&mut new_img, &rotated, size, pad_w, pad_h);
    new_img
}

/// 画像をリサイズ・回転し、正方形に整形したYOLO入力データを生成します。画像の一部を拡大し，余白に配置することができます。
///
/// # Args
///
/// * `img` - リサイズと回転を行う画像
/// * `size` - リサイズ後の画像のサイズ
/// * `rotate_angle` - 回転させる角度
/// * `rotate_en` - 画像を回転させるか。事前に回転させている場合はfalseを指定してください
/// * `crop_x` - 切り取り位置のx座標 (Noneを指定すると画像中央になります)
/// * `crop_y` - 切り取り位置のy座標 (Noneを指定すると画像中央になります)
/// * `crop_w` - 切り取り幅
/// * `crop_h` - 切り取り高さ
///
/// # Return
///
/// * リサイズ、回転、パディングを行った画像のピクセルデータ
pub fn letterbox_with_patial_enlargement(
    img: &DynamicImage,
    size: u32,
    rotate_angle: u32,
    rotate_en: bool,
    crop_x: Option<u32>,
    crop_y: Option<u32>,
    crop_w: u32,
    crop_h: u32,
) -> Vec<i16> {
    let resized = img.resize(size, size, FilterType::Nearest);
    let rotated = rotate_img(&resized, if rotate_en { rotate_angle } else { 0 });

    let pad_w = rotated.width().abs_diff(size);
    let pad_h = rotated.height().abs_diff(size);

    let mut new_img = vec![0; (size * size * 4) as usize];
    place_pixels(&mut new_img, &rotated, size, 0, 0);

    let crop_x = crop_x.unwrap_or((img.width() - crop_w) / 2);
    let crop_y = crop_y.unwrap_or((img.height() - crop_h) / 2);

    let (side_w, side_h) = match rotate_angle {
        90 | 270 => (pad_w, size),
        _ => (size, pad_h),
    };

    let crop = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
    let crop_resized = crop.resize(side_w, side_h, FilterType::Gaussian);
    place_pixels(&mut new_img, &crop_resized, size, size - pad_w, 0);

    new_img
}

/// 画像をリサイズ・回転し、正方形に整形します。
///
/// # Args
///
/// * `img` - リサイズと回転を行う画像
/// * `size` - リサイズ後の画像のサイズ
/// * `rotate_angle` - 回転させる角度
///
/// # Return
///
/// * リサイズ、回転、パディングを行ったRGB画像
pub fn letterbox_img(img: &DynamicImage, size: u32, rotate_angle: u32) -> RgbImage {
    let resized = img.resize(size, size, FilterType::Nearest);
    let rotated = rotate_img(&resized, rotate_angle);

    let pad_w = rotated.width().abs_diff(size) / 2;
    let pad_h = rotated.height().abs_diff(size) / 2;

    let mut new_img = RgbImage::new(size, size);
    for (x, y, &pixel) in rotated.to_rgb8().enumerate_pixels() {
        new_img.put_pixel(x + pad_w, y + pad_h, pixel);
    }
    new_img
}

/// 画像をリサイズ・回転し、正方形に整形します。画像の一部を拡大し，余白に配置することができます。
///
/// # Args
///
/// * `img` - リサイズと回転を行う画像
/// * `size` - リサイズ後の画像のサイズ
/// * `rotate_angle` - 回転させる角度
/// * `rotate_en` - 画像を回転させるか。事前に回転させている場合はfalseを指定してください
/// * `crop_x` - 切り取り位置のx座標 (Noneを指定すると画像中央になります)
/// * `crop_y` - 切り取り位置のy座標 (Noneを指定すると画像中央になります)
/// * `crop_w` - 切り取り幅
/// * `crop_h` - 切り取り高さ
///
/// # Return
///
/// * リサイズ、回転、パディングを行った画像
pub fn letterbox_img_with_patial_enlargement(
    img: &DynamicImage,
    rotate_angle: u32,
    rotate_en: bool,
    crop_x: Option<u32>,
    crop_y: Option<u32>,
    crop_w: u32,
    crop_h: u32,
) -> RgbImage {
    let rotated = rotate_img(&img, if rotate_en { rotate_angle } else { 0 });
    let size = u32::max(rotated.width(), rotated.height());

    let pad_w = rotated.width().abs_diff(size);
    let pad_h = rotated.height().abs_diff(size);

    let mut new_img = RgbImage::new(size, size);
    for (x, y, &pixel) in rotated.to_rgb8().enumerate_pixels() {
        new_img.put_pixel(x, y, pixel);
    }

    let crop_x = crop_x.unwrap_or((rotated.width() - crop_w) / 2);
    let crop_y = crop_y.unwrap_or((rotated.height() - crop_h) / 2);

    let (side_w, side_h) = match rotate_angle {
        90 | 270 => (pad_w, size),
        _ => (size, pad_h),
    };

    let crop = rotated.crop_imm(crop_x, crop_y, crop_w, crop_h);
    let crop_resized = crop.resize(side_w, side_h, FilterType::Gaussian);
    for (x, y, &pixel) in crop_resized.to_rgb8().enumerate_pixels() {
        new_img.put_pixel(x + (size - pad_w), y, pixel);
    }
    new_img
}

const COLORS: [[u8; 3]; 10] = [
    [255, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
    [14, 23, 50],
    [28, 105, 80],
    [190, 159, 53],
    [46, 194, 148],
    [242, 30, 131],
    [97, 101, 198],
    [115, 11, 87],
];

/// 画像上に線を描画します。
///
/// # Args
///
/// * `img` - 線を描画する画像 (in-place)
/// * `x1`, `y1`, `x2`, `y2` - 線の始点と終点の座標
/// * `thickness` - 線の太さ
/// * `color` - 線の色
fn draw_line(
    img: &mut image::RgbImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    thickness: f32,
    color: image::Rgb<u8>,
) {
    let (bx, by) = (x1 - (thickness / 2.).floor(), y1 - (thickness / 2.).floor());

    let (w, h) = if x1 == x2 {
        (thickness, (y2 - y1).abs() + thickness)
    } else {
        ((x2 - x1).abs() + thickness, thickness)
    };

    let rect = Rect::at(bx as i32, by as i32).of_size(w as u32, h as u32);
    draw_filled_rect_mut(img, rect, color);
}

/// 画像上に矩形を描画します。
///
/// # Args
///
/// * `img` - 矩形を描画する画像 (in-place)
/// * `x1`, `y1`, `x2`, `y2` - 矩形の左上と右下の座標
/// * `thickness` - 線の太さ
/// * `color` - 線の色
fn draw_rect(
    img: &mut image::RgbImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    thickness: f32,
    color: image::Rgb<u8>,
) {
    draw_line(img, x1, y1, x1, y2, thickness, color);
    draw_line(img, x1, y2, x2, y2, thickness, color);
    draw_line(img, x1, y1, x2, y1, thickness, color);
    draw_line(img, x2, y1, x2, y2, thickness, color);
}

/// 画像上にラベルを描画します。
///
/// # Args
///
/// * `img` - ラベルを描画する画像 (in-place)
/// * `x1`, `y1` - ラベルの左上の座標
/// * `line_thickness` - ラベルの枠線の太さ
/// * `bg_color` - ラベルの背景色
/// * `font` - ラベルのフォント
/// * `font_size` - ラベルのフォントサイズ
/// * `text` - ラベルに表示するテキスト
fn draw_label(
    img: &mut image::RgbImage,
    x1: f32,
    y1: f32,
    line_thickness: f32,
    bg_color: image::Rgb<u8>,
    font: &Font,
    font_size: f32,
    text: &str,
) {
    let label_h = font_size;
    let dx1 = x1 - (line_thickness / 2.).floor();
    let label_y = y1 - label_h;

    let pad = 6.;
    let scale = Scale::uniform(label_h);
    let (text_w, _) = text_size(scale, &font, &text);
    let v_metrics = font.v_metrics(scale);
    let text_h = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;

    let rect = Rect::at(dx1 as i32, label_y as i32)
        .of_size((text_w as f32 + pad * 2.) as u32, label_h as u32);
    draw_filled_rect_mut(img, rect, bg_color);

    let text_y = label_y + (label_h - text_h) / 2.;

    let text_color = if (bg_color[0] as i32 + bg_color[1] as i32 + bg_color[2] as i32) < 382 {
        Rgb([255u8, 255, 255])
    } else {
        Rgb([0u8, 0, 0])
    };
    draw_text_mut(
        img,
        text_color,
        (dx1 + pad) as i32,
        text_y as i32,
        scale,
        &font,
        &text,
    );
}

/// 画像上にバウンディングボックスとラベルを描画します。
///
/// # Args
///
/// * `img` - バウンディングボックスとラベルを描画する画像 (in-place)
/// * `d_result` - 検出結果の配列
/// * `font_size` - ラベルのフォントサイズ
/// * `line_thickness` - バウンディングボックスの線の太さ
pub fn draw_bbox(
    img: &mut image::RgbImage,
    d_result: &[DetectionData],
    font_size: f32,
    line_thickness: f32,
) {
    let font = Vec::from(include_bytes!("RobotoMono.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();
    let mut sorted = d_result.to_vec();
    sorted.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

    for d in sorted.iter() {
        let color: image::Rgb<u8> = *image::Rgb::from_slice(&COLORS[d.class as usize]);

        let x1 = d.x1.round();
        let y1 = d.y1.round();
        let x2 = d.x2.round();
        let y2 = d.y2.round();

        draw_rect(img, x1, y1, x2, y2, line_thickness, color);

        let text = format!("{}: {:.2}", d.class, d.confidence);
        draw_label(img, x1, y1, line_thickness, color, &font, font_size, &text);
    }
}
