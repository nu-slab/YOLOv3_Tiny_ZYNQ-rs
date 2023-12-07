use image::imageops::FilterType;
use image::{DynamicImage, Pixel, Rgb, RgbImage};

use crate::detection_result::DetectionData;

use imageproc::drawing::draw_filled_rect_mut;
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};

pub fn rotate_img(img: &DynamicImage, angle: u32) -> DynamicImage {
    match angle {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img.clone(),
    }
}

pub fn letterbox(img: &DynamicImage, size: u32, rotate_angle: u32) -> Vec<i16> {
    let resized = img.resize(size, size, FilterType::Nearest);
    let rotated = rotate_img(&resized, rotate_angle);

    let pad_w = rotated.width().abs_diff(size) / 2;
    let pad_h = rotated.height().abs_diff(size) / 2;

    let mut new_img = vec![0; (size * size * 4) as usize];
    for (x, y, pixel) in rotated.to_rgb8().enumerate_pixels() {
        let base_addr = 4 * (x + pad_w + (y + pad_h) * size) as usize;
        new_img[base_addr] = i16::from(pixel[0]);
        new_img[base_addr + 1] = i16::from(pixel[1]);
        new_img[base_addr + 2] = i16::from(pixel[2]);
        // new_img[base_addr + 3] = 0x7FFF; // for Debug
    }
    new_img
}

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

    let rect =
        Rect::at(bx as i32, by as i32).of_size(w as u32,h as u32);
    draw_filled_rect_mut(img, rect, color);
}

fn draw_rect(img: &mut image::RgbImage, x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, color: image::Rgb<u8>) {
    draw_line(img, x1, y1, x1, y2, thickness, color);
    draw_line(img, x1, y2, x2, y2, thickness, color);
    draw_line(img, x1, y1, x2, y1, thickness, color);
    draw_line(img, x2, y1, x2, y2, thickness, color);
}

fn draw_label(img: &mut image::RgbImage, x1: f32, y1: f32, line_thickness: f32, bg_color: image::Rgb<u8>, font: &Font, font_size: f32, text: &str) {
    let label_h = font_size;
    let dx1 = x1 - (line_thickness / 2.).floor();
    let label_y = y1 - label_h;

    let pad = 6.;
    let scale = Scale::uniform(label_h);
    let (text_w, _) = text_size(scale, &font, &text);
    let v_metrics = font.v_metrics(scale);
    let text_h = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;

    let rect = Rect::at(dx1 as i32, label_y as i32).of_size((text_w as f32 + pad * 2.) as u32, label_h as u32);
    draw_filled_rect_mut(img, rect, bg_color);

    let text_y = label_y + (label_h - text_h) / 2.;

    let text_color = if (bg_color[0] as i32 + bg_color[1] as i32 + bg_color[2] as i32) < 382 {
        Rgb([255u8, 255, 255])
    } else {
        Rgb([0u8, 0, 0])
    };
    draw_text_mut(img, text_color, (dx1 + pad) as i32, text_y as i32 , scale, &font, &text);
}

pub fn draw_bbox(img: &mut image::RgbImage, d_result: &[DetectionData], font_size: f32, line_thickness: f32) {
    let font = Vec::from(include_bytes!("RobotoMono.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();

    for d in d_result.iter() {
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
