
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
    let (x1, x2) = if a.x1 < b.x1 {
        (b.x1, a.x1)
    }
    else {
        (a.x1, b.x1)
    };

    let (y1, y2) = if a.y1 < b.y1 {
        (b.y1, a.y1)
    }
    else {
        (a.y1, b.y1)
    };

    let (xx1, xx2) = if b.x2 < a.x2 {
        (b.x2, a.x2)
    }
    else {
        (a.x2, b.x2)
    };

    let (yy1, yy2) = if b.y2 < a.y2 {
        (b.y2, a.y2)
    }
    else {
        (a.y2, b.y2)
    };

    if x1 >= xx1 || y1 >= yy1 {
        return 0.0;
    }
    let area1 = (xx1 - x1) * (yy1 - y1);
    let area2 = (xx2 - x2) * (yy2 - y2);
    area1 as f32 / area2 as f32
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
    let mut sorted_bb = bb.to_vec();
    sorted_bb.sort_by(|a, b| (-a.confidence).partial_cmp(&(-b.confidence)).unwrap());

    for ib in 0..sorted_bb.len() {
        for it in ((ib + 1)..sorted_bb.len()).rev() {
            if iou(&sorted_bb[ib], &sorted_bb[it]) > nms_threshold {
                sorted_bb.pop();
            }
        }
        if sorted_bb.len() == ib + 1 {
            break;
        }
    }
    sorted_bb
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
        .map(|d| nms(&d, nms_threshold))
        .flatten()
        .collect();
    new_box
}
