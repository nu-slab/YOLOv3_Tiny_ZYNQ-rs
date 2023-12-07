//! # YOLOv3-Tiny for Zynq 制御ライブラリ
//!
//! このクレートは、Zynq向けYOLOv3-Tiny IPを制御するためのRustライブラリです。
//!
//! ## 主な機能
//!
//! 1. **YOLOv3-Tinyモデルの初期化**: モデルの重みやバイアスなどのパラメータを設定します。
//! 2. **画像の物体検出**: YOLOv3-Tinyモデルを使用して画像から物体を検出します。
//! 3. **カメラ画像の物体検出**: カメラから取得した画像をリアルタイムで物体検出します。
//! 4. **後処理**: YOLOの出力を人間が理解しやすい形式に変換します。
//!
//! ## Example
//! ```
//! let wdir = "examples/weights";  // 重みファイルがあるディレクトリ
//! let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wdir, wdir)?;
//! let result = yolo.start(&test_img, 0)?;
//! ```

pub mod layer_group;
pub mod postprocess;
pub mod img_proc;
pub mod detection_result;
pub mod yolov3_tiny;

mod nms;
mod yolo;
