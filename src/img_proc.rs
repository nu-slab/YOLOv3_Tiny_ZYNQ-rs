use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};

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
