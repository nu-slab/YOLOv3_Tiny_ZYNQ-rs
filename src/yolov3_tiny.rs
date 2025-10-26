//! YOLOv3-Tiny のモデルをコントロールするモジュール

use std::path::Path;
use anyhow::{bail, ensure, Context, Result};
use image::{DynamicImage, Rgb, RgbImage};
use rusttype::Font;
use std::fs;

use crate::detection_result::DetectionData;
use crate::img_proc;
use crate::layer_group::{Activation, LayerGroup, PostProcess};
use crate::postprocess;
use crate::yolo::YoloController;
use crate::region::Region;

/// YOLOv3-Tiny のモデルをコントロールする構造体
pub struct YoloV3Tiny {
    yc: YoloController,
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
    n_regions: u32,
    trim_rate: f32,
}

impl YoloV3Tiny {
    /// 新しい `YoloV3Tiny` インスタンスを作成します。
    ///
    /// # Args
    /// * `hwinfo_path` - HW情報のパス
    /// * `yolo_hier` - YOLO階層のパス
    /// * `cls_num` - クラス数
    /// * `obj_threshold` - オブジェクトの閾値
    /// * `nms_threshold` - NMSの閾値
    /// * `weights_dir` - 重みのディレクトリ
    /// * `biases_dir` - バイアスのディレクトリ
    ///
    /// # Return
    /// * 新たな `YoloV3Tiny` インスタンス
    pub fn new<P: AsRef<Path>>(
        hwinfo_path: &str,
        yolo_hier: &str,
        cls_num: usize,
        obj_threshold: f32,
        nms_threshold: f32,
        weights_path: P,
    ) -> Result<Self> {
        let yc = YoloController::new(hwinfo_path, yolo_hier)?;

        let mut s = Self {
            yc,
            cls_num,
            obj_threshold,
            nms_threshold,
            n_regions: 3,
            trim_rate: 0.12,
        };
        s.init(weights_path)?;

        Ok(s)
    }

    /// YOLOv3-Tiny モデルを初期化します。
    ///
    /// # Args
    /// * `weights_dir` - 重みのディレクトリ
    /// * `biases_dir` - バイアスのディレクトリ
    #[rustfmt::skip]
    pub fn init<P: AsRef<Path>>(&mut self, weights_path: P) -> Result<()> {
        self.yc.layer_groups.push(LayerGroup::new(416, 416,  3,  1, 208, 208, 16,  1, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.yc.layer_groups.push(LayerGroup::new(208, 208, 16,  1, 104, 104, 32,  1, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.yc.layer_groups.push(LayerGroup::new(104, 104, 32,  1,  52,  52, 32,  2, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.yc.layer_groups.push(LayerGroup::new( 52,  52, 32,  2,  26,  26, 32,  4, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.yc.layer_groups.push(LayerGroup::new( 26,  26, 32,  4,  26,  26, 32,  8, false,  Activation::Leaky,     PostProcess::None, 2));
        self.yc.layer_groups.push(LayerGroup::new( 26,  26, 32,  1,  13,  13, 32,  8,  true, Activation::Linear,  PostProcess::MaxPool, 2));
        self.yc.layer_groups.push(LayerGroup::new( 13,  13, 32,  8,  13,  13, 32, 16, false,  Activation::Leaky,  PostProcess::MaxPool, 1));
        self.yc.layer_groups.push(LayerGroup::new( 13,  13, 32, 16,  13,  13, 32, 32, false,  Activation::Leaky,     PostProcess::None, 2));
        self.yc.layer_groups.push(LayerGroup::new( 13,  13, 32, 32,  13,  13, 32,  8, false,  Activation::Leaky,     PostProcess::None, 2));
        self.yc.layer_groups.push(LayerGroup::new( 13,  13, 32,  8,  13,  13, 32, 16, false,  Activation::Leaky,     PostProcess::None, 2));
        self.yc.layer_groups.push(LayerGroup::new( 13,  13, 32, 16,  13,  13, 32,  8, false, Activation::Linear,     PostProcess::Yolo, 2));
        self.yc.layer_groups.push(LayerGroup::new( 13,  13, 32,  8,  26,  26, 32,  4, false,  Activation::Leaky, PostProcess::Upsample, 2));
        self.yc.layer_groups.push(LayerGroup::new( 26,  26, 32, 12,  26,  26, 32,  8, false,  Activation::Leaky,     PostProcess::None, 2));
        self.yc.layer_groups.push(LayerGroup::new( 26,  26, 32,  8,  26,  26, 32,  8, false, Activation::Linear,     PostProcess::Yolo, 2));

        self.read_weights_and_biases(weights_path)
    }

    /// 重みとバイアスデータを読み込みます。
    ///
    /// # Args
    /// * `path` - 重みとバイアスデータが格納されているgzipアーカイブへのパス
    ///
    /// # 注意
    /// この関数は各レイヤーグループの重みとバイアスデータを読み込みます。データは16ビット整数として解釈されます。
    /// * ファイル名が "biases" で始まる場合、バイアスデータとして解釈されます。
    /// * ファイル名が "weights" で始まる場合、重みデータとして解釈されます。
    /// * それ以外のファイル名の場合、警告がログに出力され、そのファイルは無視されます。
    pub fn read_weights_and_biases<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.yc.read_weights_and_biases(path)
    }

    /// 入力データの処理を開始します。
    ///
    /// # Args
    /// * `input_data` - 入力データ
    ///
    /// # Return
    /// * YOLOの出力 (scale1, scale2)
    pub fn start_processing(&mut self, input_data: &[i16]) -> Result<(Vec<i16>, Vec<i16>)> {
        self.yc.layer_groups[0].inputs = Some(Vec::from(input_data));

        for grp_idx in 0..=13 {
            self.yc.start_layer_processing(grp_idx)?;

            if grp_idx == 4 || grp_idx == 8 {
                // あとで使うため，cloneする
                self.yc.layer_groups[grp_idx + 1].inputs =
                    self.yc.layer_groups[grp_idx].outputs.clone();
            } else if grp_idx == 10 {
                // レイヤ11の入力はレイヤ8
                self.yc.layer_groups[11].inputs =
                    self.yc.layer_groups[8].outputs.take();
            } else if grp_idx != 13 {
                // あとで使わないものはmoveして高速化
                self.yc.layer_groups[grp_idx + 1].inputs =
                    self.yc.layer_groups[grp_idx].outputs.take();
            }

            if grp_idx == 11 {
                // レイヤ12の入力はレイヤ11とレイヤ4をconcatしたもの
                // レイヤ11のデータはすでに上でmoveしているので，レイヤ4のデータを結合してあげる
                let output4 = self.yc.layer_groups[4]
                    .outputs
                    .take()
                    .context("layer_groups[4].outputs not set")?;

                match &mut self.yc.layer_groups[12].inputs {
                    Some(inputs) => inputs.extend(output4),
                    None => {
                        bail!("layer_groups[12].inputs not set");
                    }
                }
            }
        }

        // CNNの結果たち
        let output10 = self.yc.layer_groups[10]
            .outputs
            .take()
            .context("layer_groups[10].inputs not set")?;
        let output13 = self.yc.layer_groups[13]
            .outputs
            .take()
            .context("layer_groups[13].inputs not set")?;

        Ok((output10, output13))
    }

    /// 画像の処理を開始します。
    ///
    /// # Args
    /// * `img` - 入力画像
    /// * `rotate_angle` - 回転角度
    ///
    /// # Return
    /// * 物体検出結果
    pub fn start(&mut self, input_data: &[i16]) -> Result<Vec<DetectionData>> {
        let (yolo_out_0, yolo_out_1) = self.start_processing(input_data)?;

        let pp = postprocess::post_process(
            &yolo_out_0,
            &yolo_out_1,
            self.cls_num,
            self.obj_threshold,
            self.nms_threshold,
        );
        Ok(pp)
    }

    /// 画像の処理を開始します。
    ///
    /// # Args
    /// * `img` - 入力画像
    /// * `rotate_angle` - 回転角度
    ///
    /// # Return
    /// * 物体検出結果
    pub fn start_with_img_proc(
        &mut self,
        img: &DynamicImage,
        rotate_angle: u32,
    ) -> Result<Vec<DetectionData>> {
        let img_size = self.yc.layer_groups[0].input_width;
        let input_data = img_proc::letterbox(img, img_size, rotate_angle);

        let objs_rev = self
            .start(&input_data)?
            .iter()
            .map(|d| d.reverse_transform(img.width(), img.height(), rotate_angle, false))
            .collect();

        Ok(objs_rev)
    }

    /// 画像の処理を開始します。
    ///
    /// # Args
    /// * `img` - 入力画像
    /// * `rotate_angle` - 回転角度
    /// * `rotate_en` - 画像を回転させるか。事前に回転させている場合はfalseを指定してください
    ///
    /// # Return
    /// * 物体検出結果
    pub fn start_with_patial_enlargement(
        &mut self,
        img: &DynamicImage,
        rotate_angle: u32,
        rotate_en: bool,
        crop_x: Option<u32>,
        crop_y: Option<u32>,
        crop_w: u32,
        crop_h: u32,
        yolo_en: bool,
    ) -> Result<Vec<DetectionData>> {
        let img_size = self.yc.layer_groups[0].input_width;
        let input_data = img_proc::letterbox_with_patial_enlargement(
            img,
            img_size,
            rotate_angle,
            rotate_en,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
        );

        // YOLO の生座標を取得
        let objs_raw : Vec<_> = self.start(&input_data)?;

        // YOLOで信号機の色判定まで行う場合は座標変換して返す
        if yolo_en {
            let objs_rev = objs_raw
                .iter()
                .map(|d| d.reverse_transform(img.width(), img.height(), rotate_angle, true))
                .collect();
            return Ok(objs_rev);
        }

        // --- 画像処理で信号機判定とその信号の色判定を行う ---
        let mut validated_objs = Vec::new();

        // 信号機判定とデバッグ用に使われる画像を生成
        let resized = DynamicImage::from(
            img_proc::fast_resize(
                img.as_rgb8().unwrap(),
                img_size,
                img_size
            ));

        let rotated =
            img_proc::rotate_img(&resized,
                if rotate_en {
                    rotate_angle
                } else {
                    0
                }
            );

        let mut letterbox_img = RgbImage::new(img_size, img_size);

        for (x, y, &pixel) in rotated.to_rgb8().enumerate_pixels() {
            if x < img_size && y < img_size {
                letterbox_img.put_pixel(x, y, pixel);
            }
        }

        // 画像の一部を拡大して配置
        let crop_x = crop_x.unwrap_or((img.width() - crop_w) / 2);
        let crop_y = crop_y.unwrap_or((img.height() - crop_h) / 2);

        let pad_w_enlarge = rotated.width().abs_diff(img_size);
        let pad_h_enlarge = rotated.height().abs_diff(img_size);

        let (side_w, side_h) = match rotate_angle {
            90 | 270 => (pad_w_enlarge, img_size),
            _ => (img_size, pad_h_enlarge),
        };

        let crop = img.crop_imm(crop_x, crop_y, crop_w, crop_h);
        let crop_resized = img_proc::fast_resize(crop.as_rgb8().unwrap(), side_w, side_h);

        for (x, y, &pixel) in crop_resized.enumerate_pixels() {
            let final_x = x + (img_size - side_w);
            let final_y = y + (img_size - side_h);

            if final_x < img_size && final_y < img_size {
                letterbox_img.put_pixel(final_x, final_y, pixel);
            }
        }

        // デバッグ用の初期設定
        const DEBUG_MODE: bool = true;
        const DEBUG_OUTPUT_PATH: &str = "./debug_detection";

        let debug_output_dir =
            if DEBUG_MODE {
                Some(Path::new(DEBUG_OUTPUT_PATH))
            } else {
                None
            };
        let debug_mode = DEBUG_MODE;
        let mut debug_log: Vec<String> = Vec::new();

        // YOLOが検出したバウンディングボックス用の画像
        let mut debug_img_before =
            if debug_mode {
                letterbox_img.clone()
            } else {
                RgbImage::new(0, 0)
            };

        // トリミング後の領域表示用の画像
        let mut debug_img_after =
            if debug_mode {
                letterbox_img.clone()
            } else {
                RgbImage::new(0, 0)
            };

        let color_red = Rgb([255u8, 0, 0]);
        let color_yellow = Rgb([255u8, 255, 0]);
        let color_blue = Rgb([0, 0, 255]);
        let color_gray = Rgb([150u8, 150, 150]);
        let color_orange = Rgb([255u8, 140, 0]);
        let line_thickness = 2.0;

        let font =
            if debug_mode {
                let font_data = Vec::from(include_bytes!("RobotoMono.ttf") as &[u8]);
                Some(Font::try_from_vec(font_data).context("Failed to load font in yolov3_tiny.rs")?)
            } else {
                None
            };
        let font_size = 16.0;

        if debug_mode {
            debug_log.push(format!("--- [Debug] Start Validation ---"));
            debug_log.push(format!("YOLO detected {} objects.", objs_raw.len()));
        }

        for (i, mut d_data) in objs_raw.into_iter().enumerate() {
            let yolo_class_str = match d_data.class { 0 => "Red", 1 => "Yellow", 2 => "Blue", _ => "Other", };

            if debug_mode {
                debug_log.push(format!("[Object {}] YOLO Class: {}", i, yolo_class_str));
                // YOLOが検出したバウンディングボックスを描画
                img_proc::draw_rect(
                    &mut debug_img_before,
                    d_data.x1,
                    d_data.y1,
                    d_data.x2,
                    d_data.y2,
                    line_thickness,
                    color_gray
                );
            }

            // 信号機のみ画像処理の対象
            if d_data.class <= 2 {
                if debug_mode {
                    debug_log.push(format!("[Object {}] Raw Coords: x1={:.2}, y1={:.2}, x2={:.2}, y2={:.2}",
                        i, d_data.x1, d_data.y1, d_data.x2, d_data.y2));
                }

                // YOLOが検出したバウンディングボックスをトリミング
                let trim_rate_w = self.trim_rate * 0.5;
                let trim_rate_h = self.trim_rate;

                let trim_w: f32 = (d_data.x2 - d_data.x1) * trim_rate_w;
                let trim_h: f32 = (d_data.y2 - d_data.y1) * trim_rate_h;

                // 輝度や色相を分析する領域
                let bbox_result = Region::new(
                    (d_data.x1 + trim_w, d_data.y1 + trim_h),
                    (d_data.x2 - trim_w, d_data.y2 - trim_h),
                );

                // トリミング後の切り抜き画像保存
                if debug_mode {
                    if let Some(dir) = debug_output_dir {
                        if let Ok(bbox) = &bbox_result {
                            let x = bbox.start.0;
                            let y = bbox.start.1;
                            let w = bbox.width();
                            let h = bbox.height();

                            // トリミングする領域の座標が元画像を超えないように設定
                            let x_clip = x.min(letterbox_img.width().saturating_sub(1));
                            let y_clip = y.min(letterbox_img.height().saturating_sub(1));
                            let w_clip = if x_clip + w > letterbox_img.width() { letterbox_img.width() - x_clip } else { w };
                            let h_clip = if y_clip + h > letterbox_img.height() { letterbox_img.height() - y_clip } else { h };

                            if w_clip > 0 && h_clip > 0 {
                                let cropped_img = image::imageops::crop_imm(&letterbox_img, x_clip, y_clip, w_clip, h_clip).to_image();
                                let filename = format!("debug_obj_{}_{}_cropped.png", i, yolo_class_str); 

                                match DynamicImage::ImageRgb8(cropped_img).save(dir.join(filename)) {
                                    Ok(_) => {}, Err(e) => { debug_log.push(format!("[Object {}] Failed to save cropped img: {}", i, e)); }
                                }
                            } else {
                                debug_log.push(format!("[Object {}] Cropped size is zero (w:{}, h:{}). Skipped save.", i, w_clip, h_clip));
                            }
                        } else {
                            debug_log.push(format!("[Object {}] Cannot save crop: BBox was invalid after trim.", i));
                        }
                    }
                }

                let bbox = match bbox_result { Ok(b) => b, Err(_) => { continue; } };

                if debug_mode {
                    // トリミング後のバウンディングボックスを描画
                     img_proc::draw_rect(
                         &mut debug_img_after,
                         bbox.start.0 as f32,
                         bbox.start.1 as f32,
                         bbox.end.0 as f32,
                         bbox.end.1 as f32,
                         line_thickness,
                         color_orange
                     );
                }

                // バウンディングボックスを3等分
                // 左の領域から順に青、黄、赤
                ensure!(self.n_regions == 3, "n_regions must be 3");

                let mut regions = Vec::new();
                let region_w = bbox.width() / self.n_regions;
                let region_h = bbox.height();

                for idx in 0..self.n_regions {
                    let start_x = bbox.start.0 + idx * region_w;
                    let start_y = bbox.start.1;

                    let end_x = if idx == self.n_regions - 1 { bbox.end.0 } else { start_x + region_w };
                    let end_y = start_y + region_h;

                    let new_region = Region::new((start_x as f32, start_y as f32), (end_x as f32, end_y as f32))?;
                    regions.push(new_region);
                }

                // ピクセル走査
                let mut x = bbox.start.0;
                let mut y = bbox.start.1;

                let total_pixels = bbox.width() * bbox.height();
                if total_pixels == 0 { continue; }

                while y < bbox.end.1.into() {
                    let pixel_data = letterbox_img.get_pixel(x, y);

                    let r = pixel_data[0];
                    let g = pixel_data[1];
                    let b = pixel_data[2];

                    // RGB値
                    let r_f64 = r as f64 / 255.0;
                    let g_f64 = g as f64 / 255.0;
                    let b_f64 = b as f64 / 255.0;

                    // 輝度
                    let v_f64 = r_f64.max(g_f64).max(b_f64);

                    for region in regions.iter_mut() {
                        if region.is_in((x, y)) {
                            region.add_rgb(r_f64, g_f64, b_f64, v_f64);
                        }
                    }

                    x = x + 1;

                    if x >= bbox.end.0 {
                        x = bbox.start.0;
                        y = y + 1;
                    }
                }

                // 検証
                // 平均輝度
                let avg_brightnesses: Vec<f64> = regions.
                    iter()
                    .map(|r| r.avg_brightness())
                    .collect();

                // 3つの平均輝度の内、最大輝度とそのインデックス
                let (max_idx, max_avg_brightness) = avg_brightnesses
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .context("Regions vector is empty")?;

                // 3つの領域の中で最も明るかった領域の平均RGB値
                let (avg_r, avg_g, avg_b) = regions[max_idx].avg_rgb();

                // 3つの領域の中で最も明るかった領域の色相
                let hue = img_proc::calculate_hue(avg_r, avg_g, avg_b);

                // 2番目に明るかった領域の平均輝度
                let other_max_avg_brightness = avg_brightnesses
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != max_idx)
                    .map(|(_, &v)| v)
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0);

                // 輝度比
                let brightness_ratio = max_avg_brightness / (other_max_avg_brightness + 1e-6);

                // 判定
                const MIN_BRIGHT_RATIO: f64 = 1.05;
                const MAX_BRIGHT_RATIO: f64 = 5.0;
                const MIN_ABSOLUTE_BRIGHTNESS: f64 = 0.55;
                const RED_HUE_RANGE: (f64, f64) = (320.0, 360.0);
                const YELLOW_HUE_RANGE: (f64, f64) = (20.0, 40.0);
                const BLUE_HUE_RANGE: (f64, f64) = (160.0, 200.0);

                // 色相チェック関数
                fn is_hue_in_range(hue: f64, range: (f64, f64)) -> bool {
                    if range.0 > range.1 {
                        hue >= range.0 || hue <= range.1
                    } else {
                        hue >= range.0 && hue <= range.1
                    }
                }

                let is_valid_hue = match max_idx {
                    0 => is_hue_in_range(hue, BLUE_HUE_RANGE),
                    1 => is_hue_in_range(hue, YELLOW_HUE_RANGE),
                    _ => is_hue_in_range(hue, RED_HUE_RANGE),
                };

                let is_traffic_light = brightness_ratio > MIN_BRIGHT_RATIO
                                        && brightness_ratio < MAX_BRIGHT_RATIO
                                        && *max_avg_brightness > MIN_ABSOLUTE_BRIGHTNESS
                                        && is_valid_hue;

                // 結果の処理 & デバッグログ
                if debug_mode {
                    let (range_str, range_val) = match max_idx {
                        0 => ("BLUE_RANGE", BLUE_HUE_RANGE),
                        1 => ("YELLOW_RANGE", YELLOW_HUE_RANGE),
                        _ => ("RED_RANGE", RED_HUE_RANGE),
                    };

                    debug_log.push(format!("[Object {}] Avg Brightness [B:{:.2}, Y:{:.2}, R:{:.2}] (Max:{:.2})",
                                    i,
                                    avg_brightnesses.get(0).unwrap_or(&0.0),
                                    avg_brightnesses.get(1).unwrap_or(&0.0),
                                    avg_brightnesses.get(2).unwrap_or(&0.0),
                                    *max_avg_brightness
                    ));

                    debug_log.push(format!(
                        "[Object {}] Hue: {:.1} degrees (Threshold: {} [{:.1}, {:.1}])",
                        i,
                        hue,
                        range_str,
                        range_val.0,
                        range_val.1
                    ));

                    debug_log.push(format!("[Object {}] Bright Ratio: {:.2} (Threshold: {} < x < {})",
                                    i,
                                    brightness_ratio,
                                    MIN_BRIGHT_RATIO,
                                    MAX_BRIGHT_RATIO
                    ));

                    debug_log.push(format!("[Object {}] Max Avg Brightness: {:.2} (Threshold: {} < x)",
                                    i,
                                    *max_avg_brightness,
                                    MIN_ABSOLUTE_BRIGHTNESS
                    ));
                }

                if is_traffic_light {
                    let class = match max_idx {
                        0 => 2, // Blue
                        1 => 1, // Yellow
                        _ => 0, // Red
                    };
                    d_data.class = class;

                    if debug_mode {
                        let class_str = match class { 0 => "Red", 1 => "Yellow", _ => "Blue" };

                        debug_log.push(format!("[Object {}] Validation OK: Detected as {}", i, class_str));

                        let color = match class { 0 => color_red, 1 => color_yellow, _ => color_blue };
                        let text = format!("{}: {:.2}", class_str, d_data.confidence);

                        if let Some(font_ref) = &font {
                            img_proc::draw_rect(&mut debug_img_before, d_data.x1, d_data.y1, d_data.x2, d_data.y2, line_thickness, color);
                            img_proc::draw_label(&mut debug_img_before, d_data.x1, d_data.y1, line_thickness, color, font_ref, font_size, &text);

                            img_proc::draw_rect(&mut debug_img_after, d_data.x1, d_data.y1, d_data.x2, d_data.y2, line_thickness, color);
                            img_proc::draw_label(&mut debug_img_after, d_data.x1, d_data.y1, line_thickness, color, font_ref, font_size, &text);
                        }
                    }
                    validated_objs.push(d_data);

                } else {
                    if debug_mode { debug_log.push(format!("[Object {}] Validation NG: Not a traffic light. REJECTED.", i)); }
                }
            } else {
                validated_objs.push(d_data);
            }
        }

        // デバッグファイルの保存
        if let Some(dir) = debug_output_dir {
            if !dir.exists() {
                std::fs::create_dir_all(dir)?;
            }

            DynamicImage::ImageRgb8(debug_img_before).save(dir.join("debug_yolo_detection.png"))?;
            DynamicImage::ImageRgb8(debug_img_after).save(dir.join("debug_ip_trimming.png"))?;

            let log_content = debug_log.join("\n");

            fs::write(dir.join("debug_validation_log.txt"), log_content)?;
        }

        let final_objs = validated_objs
            .iter()
            .map(|d| d.reverse_transform(img.width(), img.height(), rotate_angle, true))
            .collect();

        Ok(final_objs)
    }
}
