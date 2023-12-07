# YOLOv3-Tiny for Zynq 制御ライブラリ

[Zynq向けYOLOv3-Tiny IP](https://github.com/nu-slab/Kria-YOLO-HW)を制御するためのRustライブラリです。

[UMV-HardwareTools](https://github.com/nu-slab/UMV-HardwareTools) が出力するハードウェア構成情報ファイルを利用します。

## Usage

```shell
cargo add --git git@github.com:nu-slab/YOLOv3_Tiny_ZYNQ-rs.git
```

## Example

- 物体検出

```Rust
let wdir = "examples/weights";  // 重みファイルがあるディレクトリ
let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wdir, wdir)?;
let result = yolo.start(&test_img, 0)?;
```

- バウンディングボックスのプロット

```Rust
// BBox描画のためDynamicImageをRGB画像に変換
let mut rgb_img = test_img.to_rgb8();
draw_bbox(&mut rgb_img, &result, 20., 6.);
```
