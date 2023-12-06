use image::imageops::FilterType;
use image::{DynamicImage, RgbImage, Pixel};

use crate::thick_xiaolin_wu::draw_line;
use crate::utils::DetectionData;

pub fn rotate_img(img: DynamicImage, angle: u32) -> DynamicImage {
    match angle {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img,
    }
}

pub fn letterbox(img: DynamicImage, size: u32, rotate_angle: u32) -> Vec<i16> {
    let resized = img.resize(size, size, FilterType::Nearest);
    let rotated = rotate_img(resized, rotate_angle);

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

pub fn letterbox_img(img: DynamicImage, size: u32, rotate_angle: u32) -> RgbImage {
    let resized = img.resize(size, size, FilterType::Nearest);
    let rotated = rotate_img(resized, rotate_angle);

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

fn draw_rect(img: &mut image::RgbImage, x1: f32, y1: f32, x2: f32, y2: f32, color: image::Rgb<u8>) {
    let thickness = 3.;
    draw_line(img, x1, y1, x1, y2, thickness, color);
    draw_line(img, x1, y2, x2, y2, thickness, color);
    draw_line(img, x1, y1, x2, y1, thickness, color);
    draw_line(img, x2, y1, x2, y2, thickness, color);
}

pub fn draw_bbox(img: &mut image::RgbImage, d_result: &[DetectionData]) {
    for d in d_result.iter() {
        let color: image::Rgb<u8> = *image::Rgb::from_slice(&COLORS[d.class as usize]);
        draw_rect(img, d.x1, d.y1, d.x2, d.y2, color);
    }
}
