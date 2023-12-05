use anyhow::Result;
use image::Pixel;
use std::time::Instant;

use tiny_yolo_v3_zynq_rs::img_proc::letterbox_img;
use tiny_yolo_v3_zynq_rs::yolo::YoloV3Tiny;

const COLORS: [[u8; 3]; 7] = [
    [0, 0, 255],
    [0, 255, 255],
    [255, 0, 0],
    [50, 14, 23],
    [80, 28, 105],
    [53, 190, 159],
    [148, 46, 194],
];

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

    for d in result.iter() {
        let color: image::Rgb<u8> = *image::Rgb::from_slice(&COLORS[d.class as usize]);
        draw_rect(&mut lb_img, d.x1, d.y1, d.x2, d.y2, color);
    }
    lb_img.save("out.png")?;

    Ok(())
}

// --- Drawing ---

fn draw_rect(img: &mut image::RgbImage, x1: f32, y1: f32, x2: f32, y2: f32, color: image::Rgb<u8>) {
    let thickness = 3.;
    draw_line(img, x1, y1, x1, y2, thickness, color);
    draw_line(img, x1, y2, x2, y2, thickness, color);
    draw_line(img, x1, y1, x2, y1, thickness, color);
    draw_line(img, x2, y1, x2, y2, thickness, color);
}

// A simple extension to Xiaolin Wu's line drawing algorithm to draw thick lines.
// Adapted from https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
// ----
// Transelated from https://github.com/jambolo/thick-xiaolin-wu (c) 2022 John Bolton
// 2023 Shibata Lab.

/// Draw RGB pixel with alpha
fn draw_pixel(img: &mut image::RgbImage, x: i32, y: i32, alpha: f32, color: image::Rgb<u8>) {
    if x < 0 || y < 0 || (img.width() as i32) <= x || (img.height() as i32) <= y {
        return;
    }
    let pix = img.get_pixel(x as u32, y as u32);
    let new_r = (f32::from(color[0]) * alpha + f32::from(pix[0]) * (1. - alpha)).min(255.) as u8;
    let new_g = (f32::from(color[1]) * alpha + f32::from(pix[1]) * (1. - alpha)).min(255.) as u8;
    let new_b = (f32::from(color[2]) * alpha + f32::from(pix[2]) * (1. - alpha)).min(255.) as u8;

    img.put_pixel(x as u32, y as u32, image::Rgb::from([new_r, new_g, new_b]));
}

/// draw thick line
pub fn draw_line(
    img: &mut image::RgbImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: image::Rgb<u8>,
) {
    let mut x0 = x0;
    let mut y0 = y0;
    let mut x1 = x1;
    let mut y1 = y1;

    // Ensure positive integer values for thickness
    let thickness = if thickness < 1. { 1. } else { thickness };

    // steep means that m > 1
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    // If steep, then x and y must be swapped because the width is fixed in the y direction and that won't work if
    // dx < dy. Note that they are swapped again when plotting.
    if steep {
        (x0, y0) = (y0, x0);
        (x1, y1) = (y1, x1);
    }

    // Swap endpoints to ensure that dx > 0
    if x1 < x0 {
        (x0, x1) = (x1, x0);
        (y0, y1) = (y1, y0);
    }

    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx > 0. { dy / dx } else { 1. };

    let wf = thickness * (1. + gradient * gradient).sqrt();
    let wi = wf as i32;

    let xend = x0.round();
    let yend = y0 - (wf - 1.) * 0.5 + gradient * (xend - x0);
    let xgap = 1. - (x0 + 0.5 - xend);
    let xpxl1 = xend as i32; // this will be used in the main loop
    let ypxl1 = yend.floor() as i32;
    let fpart = yend - yend.floor();
    let rfpart = 1. - fpart;

    if steep {
        draw_pixel(img, ypxl1, xpxl1, rfpart * xgap, color);
        for i in 1..wi {
            draw_pixel(img, ypxl1 + i, xpxl1, 1., color);
        }
        draw_pixel(img, ypxl1 + wi, xpxl1, fpart * xgap, color);
    } else {
        draw_pixel(img, xpxl1, ypxl1, rfpart * xgap, color);
        for i in 1..wi {
            draw_pixel(img, xpxl1, ypxl1 + i, 1., color);
        }
        draw_pixel(img, xpxl1, ypxl1 + wi, fpart * xgap, color);
    }

    let mut intery = yend + gradient; // first y-intersection for the main loop

    // Handle second endpoint
    let xend = x1.round();
    let yend = y1 - (wf - 1.) * 0.5 + gradient * (xend - x1);
    let xgap = 1. - (x1 + 0.5 - xend);
    let xpxl2 = xend as i32; // this will be used in the main loop
    let ypxl2 = yend.floor() as i32;
    let fpart = yend - yend.floor();
    let rfpart = 1. - fpart;

    if steep {
        draw_pixel(img, ypxl2, xpxl2, rfpart * xgap, color);
        for i in 1..wi {
            draw_pixel(img, ypxl2 + i, xpxl2, 1., color);
        }
        draw_pixel(img, ypxl2 + wi, xpxl2, fpart * xgap, color);
    } else {
        draw_pixel(img, xpxl2, ypxl2, rfpart * xgap, color);
        for i in 1..wi {
            draw_pixel(img, xpxl2, ypxl2 + i, 1., color);
        }
        draw_pixel(img, xpxl2, ypxl2 + wi, fpart * xgap, color);
    }

    // main loop
    if steep {
        for x in xpxl1 + 1..xpxl2 {
            let fpart = intery - intery.floor();
            let rfpart = 1. - rfpart;
            let y = intery.floor() as i32;
            draw_pixel(img, y, x, rfpart * xgap, color);
            for i in 1..wi {
                draw_pixel(img, y + i, x, 1., color);
            }
            draw_pixel(img, y + wi, x, fpart * xgap, color);
            intery += gradient;
        }
    } else {
        for x in xpxl1 + 1..xpxl2 {
            let fpart = intery - intery.floor();
            let rfpart = 1. - rfpart;
            let y = intery.floor() as i32;
            draw_pixel(img, x, y, rfpart * xgap, color);
            for i in 1..wi {
                draw_pixel(img, x, y + i, 1., color);
            }
            draw_pixel(img, x, y + wi, fpart * xgap, color);
            intery += gradient;
        }
    }
}
