//! YOLO (You Only Look Once) 物体検出アルゴリズムの出力を後処理するためのモジュール

use crate::detection_result::DetectionData;
use crate::nms::nms_process;

const ANCHOR_BOX_NUM: usize = 3;

/// `fix2float`関数は、符号あり[8bits].[8bits]の固定小数点数をf32型の浮動小数点数に変換します
///
/// # Args
/// * `input` - f32型に変換するi16型の固定小数点数
///
/// # Return
/// * 入力値を2の8乗で除算したf32型の浮動小数点数
fn fix2float(input: i16) -> f32 {
    input as f32 / 2f32.powi(8)
}

/// ch_reorder関数は、与えられた配列を再配置します
///
/// # Args
/// * `arr` - 再配置するf32型の配列
/// * `grid_num` - グリッドの数（配列の再配置に使用）
///
/// # Return
/// * 再配置されたf32型のベクトル
fn ch_reorder(arr: &[f32], grid_num: usize) -> Vec<f32> {
    let mut reorder: Vec<f32> = vec![];
    for i in 0..grid_num * grid_num {
        for j in 0..8 {
            for k in 0..32 {
                reorder.push(arr[(grid_num * grid_num * 32) * j + 32 * i + k]);
            }
        }
    }
    reorder
}

/// `ch_reshape`関数は、与えられた配列を再形成します
///
/// # Args
/// * `reorder_arr` - 再形成するf32型の配列
/// * `grid_num` - グリッドの数（配列の再形成に使用）
/// * `cls_num` - クラスの数（配列の再形成に使用）
///
/// # Return
/// * 再形成された2つのf32型のベクトル (reshape, class)
fn ch_reshape(reorder_arr: &[f32], grid_num: usize, cls_num: usize) -> (Vec<f32>, Vec<f32>) {
    let mut reshape = vec![0.; grid_num * grid_num * 18];
    let mut class = vec![0.; grid_num * grid_num * ANCHOR_BOX_NUM * cls_num];
    let mut cnt_cls = 0;

    for i in (0..grid_num * grid_num * 18).step_by(18) {
        for j in 0..ANCHOR_BOX_NUM {
            for k in 0..cls_num {
                class[cnt_cls + j * cls_num + k] = reorder_arr[(i / 18) * 256 + 85 * j + 5 + k];
            }
        }
        cnt_cls += ANCHOR_BOX_NUM * cls_num;

        for index in 0..18 {
            let base_index = (i / 18) * 256;
            let reorder_index = base_index + 85 * (index / 6) + (index % 6);
            let offset = if index % 6 == 5 { 1 } else { 0 };
            reshape[i + index] = reorder_arr[reorder_index + offset];
        }
    }
    (reshape, class)
}

/// get_anchor_box関数は、アンカーボックスの値を計算します
///
/// # Args
/// * `reshape` - アンカーボックスの値を計算するためのf32型のベクトル
/// * `grid_num` - グリッドの数（アンカーボックスの計算に使用）
/// * `anchor_box` - アンカーボックスの初期値
fn get_anchor_box(reshape: &mut [f32], grid_num: usize, anchor_box: [[f32; 2]; 3]) {
    let grid_width = 416.0 / grid_num as f32;
    let mut w_cnt = 0.;
    let mut h_cnt = 0.;
    for i in (0..grid_num * grid_num * 18).step_by(18) {
        for (j, ab) in anchor_box.iter().enumerate() {
            let idx = i + 6 * j;
            reshape[idx] = grid_width * w_cnt + grid_width * reshape[idx]; //rm-sigmoid
            reshape[idx + 1] = grid_width * h_cnt + grid_width * reshape[idx + 1]; //rm-sigmoid
            reshape[idx + 2] = ab[0] * (reshape[idx + 2]).exp();
            reshape[idx + 3] = ab[1] * (reshape[idx + 3]).exp();
        }
        w_cnt += 1.;
        if w_cnt == (grid_num as f32) {
            w_cnt = 0.;
            h_cnt += 1.;
        }
    }
}

/// `get_cls_id`関数は、クラスIDを取得します
///
/// # Args
/// * `cls_concat` - クラスIDを取得するためのf32型の配列
/// * `idx` - クラスIDを取得するためのインデックス
/// * `cls_num` - クラスの数
///
/// # Return
/// * 最大の値を持つ要素のクラスID
fn get_cls_id(cls_concat: &[f32], idx: usize, cls_num: usize) -> u8 {
    let ccnt = idx * cls_num;
    ((ccnt..ccnt + cls_num)
        .max_by(|&a, &b| cls_concat[a].partial_cmp(&cls_concat[b]).unwrap())
        .unwrap()
        - ccnt) as u8
}

/// get_objs関数は、物体を検出します
///
/// # Args
/// * grid_concat - 物体検出を行うためのf32型の配列
/// * cls_concat - 物体検出を行うためのf32型の配列
/// * cls_num - クラスの数
///
/// # Return
/// * 検出された物体を表すDetectionDataのベクトル
fn get_objs(grid_concat: &[f32], cls_concat: &[f32], cls_num: usize) -> Vec<DetectionData> {
    grid_concat[..(13 * 13 + 26 * 26) * 18]
        .chunks(18 / ANCHOR_BOX_NUM)
        .enumerate()
        .flat_map(|(idx, yolo_result)| {
            DetectionData::new_from_yolo(yolo_result, get_cls_id(cls_concat, idx, cls_num))
        })
        .collect()
}

/// `post_process`関数は、YOLOの出力から物体検出を行います
///
/// # Args
/// * `yolo_out_0` - YOLOの出力
/// * `yolo_out_1` - YOLOの別の出力
/// * `cls_num` - クラスの数
/// * `obj_threshold` - 物体検出の閾値
/// * `nms_threshold` - 非最大抑制（NMS）の閾値
///
/// # Return
/// * 検出された物体を表すDetectionDataのベクトル
///
/// このベクトルは、物体検出の結果を表すデータ構造を含みます
/// 各DetectionDataは、検出された物体のクラスID、信頼度スコア、およびバウンディングボックスの座標を含みます
pub fn post_process(
    yolo_out_0: &[i16],
    yolo_out_1: &[i16],
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
) -> Vec<DetectionData> {
    // i16 >> f32
    let arr13: Vec<f32> = yolo_out_0.iter().map(|&val| fix2float(val)).collect();
    let arr26: Vec<f32> = yolo_out_1.iter().map(|&val| fix2float(val)).collect();

    //channel reorder
    //8*13*13*32 >> 13*13*256
    //8*26*26*32 >> 13*13*256
    let reorder13 = ch_reorder(&arr13, 13);
    let reorder26 = ch_reorder(&arr26, 26);

    //channel reshape 256ch >> 255ch
    //13*13*256 >> 13*13*255
    //26*26*256 >> 26*26*255
    let (mut reshape13, class13) = ch_reshape(&reorder13, 13, cls_num);
    let (mut reshape26, class26) = ch_reshape(&reorder26, 26, cls_num);

    //(座標x,y) (大きさw,h) (物体確率) (class確率80)
    //2+2+1+80 = 85
    //85 * 3(anchorBOXの数) = 255
    //13*13*255, 26*26*255
    //座標と大きさを計算,確率はそのまま
    //[[[23,27], [37,58], [81,82]], [[81,82], [135,169], [344,319]]]
    let anchor_box_13 = [[81., 82.], [135., 169.], [344., 319.]];
    let anchor_box_26 = [[23., 27.], [37., 58.], [81., 82.]];
    get_anchor_box(&mut reshape13, 13, anchor_box_13);
    get_anchor_box(&mut reshape26, 26, anchor_box_26);

    // 13*13検出と26*26検出を結合
    // 13*13*255, 26*26*255 >> (13*13+26*26)*255
    let mut grid_concat = reshape13;
    grid_concat.extend(reshape26);
    let mut cls_concat = class13;
    cls_concat.extend(class26);

    // ディテクション結果を抽出
    let nms_boxes = get_objs(&grid_concat, &cls_concat, cls_num);

    // NMS を適用
    nms_process(&nms_boxes, cls_num, obj_threshold, nms_threshold)
}
