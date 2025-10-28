//! 分割したバウンディングボックスに関する画像処理モジュール

use anyhow::{ensure, Result};

pub struct Region {
    pub start: (u32, u32),
    pub end: (u32, u32),
    total_brightness: f64,
    total_r: f64,
    total_g: f64,
    total_b: f64,
    pixel_count: u32,
}

impl Region {
    pub fn new(s: (f32, f32), e: (f32, f32)) -> Result<Self> {
        let values = [s.0, s.1, e.0, e.1];

        ensure!(values.iter().all(|f| f.is_sign_positive()), "Coordinates must be positive");

        let start = (s.0.floor() as u32, s.1.floor() as u32);
        let end = (e.0.floor() as u32, e.1.floor() as u32);

        ensure!(start.0 <= end.0 && start.1 <= end.1, "Start coordinates must be less than or equal to end coordinates");

        let total_brightness = 0.0;
        let total_r = 0.0;
        let total_g = 0.0;
        let total_b = 0.0;
        let pixel_count = 0;

        Ok(Self { start, end, total_brightness, total_r, total_g, total_b, pixel_count })
    }

    pub fn width(&self) -> u32 {
        self.end.0.saturating_sub(self.start.0)
    }

    pub fn height(&self) -> u32 {
        self.end.1.saturating_sub(self.start.1)
    }

    pub fn is_in(&self, p: (u32, u32)) -> bool {
        p.0 >= self.start.0 && p.1 >= self.start.1 && p.0 < self.end.0 && p.1 < self.end.1
    }

    pub fn add_rgb(&mut self, r: f64, g: f64, b: f64, v: f64) {
        self.total_r += r;
        self.total_g += g;
        self.total_b += b;
        self.total_brightness += v;
        self.pixel_count += 1;
    }

    pub fn avg_brightness(&self) -> f64 {
        if self.pixel_count == 0 {
            0.0
        }
        else {
            self.total_brightness / self.pixel_count as f64
        }
    }

    pub fn avg_rgb(&self) -> (f64, f64, f64) {
        if self.pixel_count == 0 {
            (0.0, 0.0, 0.0)
        } else {(
            self.total_r / self.pixel_count as f64,
            self.total_g / self.pixel_count as f64,
            self.total_b / self.pixel_count as f64
        )}
    }
}
