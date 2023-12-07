use crate::detection_result::DetectionData;
use crate::nms::nms_process;

const ANCHOR_BOX_NUM: usize = 3;

fn fix2float(input: i16) -> f32 {
    input as f32 / 2f32.powi(8)
}

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

fn get_anchor_box(reshape: &mut Vec<f32>, grid_num: usize, anchor_box: [[f32; 2]; 3]) {
    let grid_width = 416.0 / grid_num as f32;
    let mut w_cnt = 0.;
    let mut h_cnt = 0.;
    for i in (0..grid_num * grid_num * 18).step_by(18) {
        for j in 0..=2 {
            let idx = i + 6 * j;
            reshape[idx + 0] = grid_width * w_cnt + grid_width * reshape[idx]; //rm-sigmoid
            reshape[idx + 1] = grid_width * h_cnt + grid_width * reshape[idx + 1]; //rm-sigmoid
            reshape[idx + 2] = anchor_box[j][0] * (reshape[idx + 2]).exp();
            reshape[idx + 3] = anchor_box[j][1] * (reshape[idx + 3]).exp();
        }
        w_cnt += 1.;
        if w_cnt == (grid_num as f32) {
            w_cnt = 0.;
            h_cnt += 1.;
        }
    }
}

fn get_cls_id(cls_concat: &[f32], idx: usize, cls_num: usize) -> u8 {
    let ccnt = idx * cls_num;
    ((ccnt..ccnt + cls_num)
        .max_by(|&a, &b| cls_concat[a].partial_cmp(&cls_concat[b]).unwrap())
        .unwrap()
        - ccnt) as u8
}

fn get_objs(grid_concat: &[f32], cls_concat: &[f32], cls_num: usize) -> Vec<DetectionData> {
    grid_concat[..(13 * 13 + 26 * 26) * 18]
        .chunks(18 / ANCHOR_BOX_NUM)
        .enumerate()
        .flat_map(|(idx, yolo_result)| {
            DetectionData::new_from_yolo(yolo_result, get_cls_id(cls_concat, idx, cls_num))
        })
        .collect()
}

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
