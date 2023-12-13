
use crate::detection_result::DetectionData;

/// 2つの検出データ間のIoU（Intersection over Union）を計算します。
///
/// # Args
/// * `a` - 検出データ1
/// * `b` - 検出データ2
///
/// # Return
/// * IoUの値（0.0から1.0の範囲）
fn iou(a: &DetectionData, b: &DetectionData) -> f32 {
    let dx = a.x2.min(b.x2) - a.x1.max(b.x1);
    let dy = a.y2.min(b.y2) - a.y1.max(b.y1);
    let inter_area = (dx * dy).max(0.);

    let area1 = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area2 = (b.x2 - b.x1) * (b.y2 - b.y1);

    inter_area / (area1 + area2 - inter_area)
}

/// Non-Maximum Suppression (NMS)を適用して、重複した検出を削除します。
///
/// # Args
/// * `bb` - 検出データの配列
/// * `nms_threshold` - NMSの閾値
///
/// # Return
/// * NMSを適用した後の検出データの配列
fn nms(bb: &[DetectionData], nms_threshold: f32) -> Vec<DetectionData> {
    let mut detections = bb.to_vec();
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = vec![];
    while !detections.is_empty() {
        let detection = detections.remove(0);
        keep.push(detection);

        detections.retain(|x| iou(&detection, x) < nms_threshold);
    }
    keep
}

/// 検出データをクラスごとに分割し、各クラスにNMSを適用します。
///
/// # Args
/// * `bb` - 検出データの配列
/// * `cls_num` - クラスの数
/// * `obj_threshold` - オブジェクト検出の閾値
/// * `nms_threshold` - NMSの閾値
///
/// # Return
/// * NMSを適用した後の検出データの配列
pub fn nms_process(
    bb: &[DetectionData],
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
) -> Vec<DetectionData> {
    // クラス別に分割
    let mut cls: Vec<Vec<DetectionData>> = vec![vec![]; cls_num];
    for &detection in bb {
        if detection.confidence > obj_threshold && detection.confidence <= 1.0 {
            cls[detection.class as usize].push(detection);
        }
    }

    // 各クラスに Non-Maximum Suppression (NMS) を適用し，重なっているBBoxの中でコンフィデンスが最大のものを集める
    let new_box: Vec<DetectionData> = cls
        .into_iter()
        .flat_map(|d| nms(&d, nms_threshold))
        .collect();
    new_box
}
