//! 物体検出の結果を処理するモジュール

use anyhow::{anyhow, Result};

/// 送られてきた生の検出結果を保持するための構造体
#[derive(Debug, Clone, Copy)]
pub struct DetectionData {
    /// クラス
    pub class: u8,
    /// バウンディングボックス左上のx
    pub x1: f32,
    /// バウンディングボックス左上のy
    pub y1: f32,
    /// バウンディングボックス右下のx
    pub x2: f32,
    /// バウンディングボックス右下のy
    pub y2: f32,
    /// コンフィデンス
    pub confidence: f32,
}

impl DetectionData {
    /// YOLOの結果から新しいDetectionDataを作成します。
    ///
    /// # Args
    ///
    /// * `yolo_result` - YOLOの結果の配列
    /// * `cls_id` - クラスID
    ///
    /// # Return
    /// * 新たなDetectionDataインスタンス
    pub fn new_from_yolo(yolo_result: &[f32], cls_id: u8) -> Result<Self> {
        // 中心座標
        let cx = yolo_result[0];
        let cy = yolo_result[1];

        // BBoxのサイズ
        let cw = yolo_result[2];
        let ch = yolo_result[3];

        let nms_box = Self {
            class: cls_id,
            x1: cx - cw / 2.,
            y1: cy - ch / 2.,
            x2: cx + cw / 2.,
            y2: cy + ch / 2.,
            confidence: yolo_result[4],
        };
        if (0. <= nms_box.x1 && nms_box.x1 <= 416.)
            && (0. <= nms_box.y1 && nms_box.y1 <= 416.)
            && (0. <= nms_box.x2 && nms_box.x2 <= 416.)
            && (0. <= nms_box.y2 && nms_box.y2 <= 416.)
        {
            Ok(nms_box)
        } else {
            Err(anyhow!("nms_box out of range: {:?}", nms_box))
        }
    }

    /// YOLOの出力した検出結果の座標を元の画像の座標系に戻します。
    ///
    /// # Args
    ///
    /// * `width` - 画像の幅
    /// * `height` - 画像の高さ
    /// * `rotate_angle` - 回転角度
    ///
    /// # Return
    /// * 新たなDetectionDataインスタンス
    pub fn reverse_transform(&self, width: u32, height: u32, rotate_angle: u32) -> Self {
        let mut new_d = *self;
        (new_d.x1, new_d.y1) =
            point_reverse_transform(width, height, rotate_angle, self.x1, self.y1);
        (new_d.x2, new_d.y2) =
            point_reverse_transform(width, height, rotate_angle, self.x2, self.y2);
        new_d
    }
}


    /// YOLOの出力した座標を元の画像の座標系に戻します。
    ///
    /// # Args
    ///
    /// * `width` - 画像の幅
    /// * `height` - 画像の高さ
    /// * `rotate_angle` - 回転角度
    /// * `x` - x座標
    /// * `y` - y座標
    ///
    /// # Return
    /// * 新たな座標 (x, y)
fn point_reverse_transform(
    width: u32,
    height: u32,
    rotate_angle: u32,
    x: f32,
    y: f32,
) -> (f32, f32) {
    let yolo_input_size = 416.;

    let (w, h) = match rotate_angle {
        90 | 270 => (height, width),
        _ => (width, height),
    };

    let wratio = yolo_input_size / w as f32;
    let hratio = yolo_input_size / h as f32;
    let ratio = f32::min(wratio, hratio);
    let nw = w as f32 * ratio;
    let nh = h as f32 * ratio;

    let pad_w = (yolo_input_size - nw) / 2.;
    let pad_h = (yolo_input_size - nh) / 2.;

    ((x - pad_w) / ratio, (y - pad_h) / ratio)
}
