//! YOLOv3-Tiny のモデルをコントロールするモジュール

use std::path::Path;

use anyhow::{bail, Context, Result};
use image::DynamicImage;

use crate::detection_result::DetectionData;
use crate::img_proc;
use crate::layer_group::{Activation, LayerGroup, PostProcess};
use crate::postprocess;
use crate::yolo::YoloController;

/// YOLOv3-Tiny のモデルをコントロールする構造体
pub struct YoloV3Tiny {
    yc: YoloController,
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
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
                self.yc.layer_groups[11].inputs = self.yc.layer_groups[8].outputs.take();
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

        let objs_rev = self
            .start(&input_data)?
            .iter()
            .map(|d| d.reverse_transform(img.width(), img.height(), rotate_angle, true))
            .collect();

        Ok(objs_rev)
    }
}
