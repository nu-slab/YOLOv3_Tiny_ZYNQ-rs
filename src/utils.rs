use log::error;

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

fn fix_to_float(input: i16) -> f32 {
    let bit_width = 16;
    let mut output = 0.0;
    for i in 0..bit_width {
        let input_shift_bit = (input >> i) & 1;
        if input_shift_bit == 1 && i < bit_width / 2 {
            output += 2.0f32.powi(-(bit_width / 2 - i));
        } else if input_shift_bit == 1 && bit_width / 2 <= i {
            if i - bit_width / 2 == 7 {
                output += 2.0f32.powi(i - bit_width / 2) * -1.0;
            } else {
                output += 2.0f32.powi(i - bit_width / 2);
            }
        }
    }
    output
}

pub fn post_process(
    yolo_out_0: &[i16],
    yolo_out_1: &[i16],
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
) -> Vec<DetectionData> {
    // i16 >> f32
    let grid13_array: Vec<f32> = yolo_out_0.iter().map(|&val| fix_to_float(val)).collect();
    let grid26_array: Vec<f32> = yolo_out_1.iter().map(|&val| fix_to_float(val)).collect();

    let grid_13 = 13;
    let grid_26 = 26;

    //channel reorder
    //8*13*13*32 >> 13*13*256
    //8*26*26*32 >> 13*13*256
    let mut grid13_array_reorder: Vec<f32> = vec![];
    for i in 0..13 * 13 {
        for j in 0..8 {
            for k in 0..32 {
                grid13_array_reorder.push(grid13_array[(grid_13 * grid_13 * 32) * j + 32 * i + k]);
            }
        }
    }
    let mut grid26_array_reorder: Vec<f32> = vec![];
    for i in 0..26 * 26 {
        for j in 0..8 {
            for k in 0..32 {
                grid26_array_reorder.push(grid26_array[(grid_26 * grid_26 * 32) * j + 32 * i + k]);
            }
        }
    }

    //channel reshape 256ch >> 255ch
    //13*13*256 >> 13*13*255
    //26*26*256 >> 26*26*255
    let anchor_box_num = 3;
    let mut grid13_array_reshape = vec![0.; grid_13 * grid_13 * 18];
    let mut grid26_array_reshape = vec![0.; grid_26 * grid_26 * 18];
    let mut grid13_array_class = vec![0.; grid_13 * grid_13 * anchor_box_num * cls_num];
    let mut grid26_array_class = vec![0.; grid_26 * grid_26 * anchor_box_num * cls_num];

    let mut cnt_cls = 0;

    for i in (0..grid_13 * grid_13 * 18).step_by(18) {
        for j in 0..anchor_box_num {
            for k in 0..cls_num {
                grid13_array_class[cnt_cls + j * cls_num + k] =
                    grid13_array_reorder[(i / 18) * 256 + 85 * j + 5 + k];
            }
        }
        cnt_cls += anchor_box_num * cls_num;

        for index in 0..18 {
            let base_index = (i / 18) * 256;
            let reorder_index = base_index + 85 * (index / 6) + (index % 6);
            let offset = if index % 6 == 5 { 1 } else { 0 };
            grid13_array_reshape[i + index] = grid13_array_reorder[reorder_index + offset];
        }
    }

    cnt_cls = 0;
    for i in (0..grid_26 * grid_26 * 18).step_by(18) {
        for ii in 0..anchor_box_num {
            for jj in 0..cls_num {
                grid26_array_class[cnt_cls + ii * cls_num + jj] =
                    grid26_array_reorder[(i / 18) * 256 + 85 * ii + 5 + jj];
            }
        }
        cnt_cls += anchor_box_num * cls_num;

        for index in 0..18 {
            let base_index = (i / 18) * 256;
            let reorder_index = base_index + 85 * (index / 6) + (index % 6);
            let offset = if index % 6 == 5 { 1 } else { 0 };
            grid26_array_reshape[i + index] = grid26_array_reorder[reorder_index + offset];
        }
    }

    //(座標x,y) (大きさw,h) (物体確率) (class確率80)
    //2+2+1+80 = 85
    //85 * 3(anchorBOXの数) = 255
    //13*13*255, 26*26*255
    //座標と大きさを計算,確率はそのまま
    //[[[23,27], [37,58], [81,82]], [[81,82], [135,169], [344,319]]]
    let anchor_box_13 = [[81., 82.], [135., 169.], [344., 319.]];
    let anchor_box_26 = [[23., 27.], [37., 58.], [81., 82.]];
    let grid_width_13 = 416.0 / 13.0;
    let grid_width_26 = 416.0 / 26.0;
    let mut w_cnt = 0.;
    let mut h_cnt = 0.;
    for i in (0..grid_13 * grid_13 * 18).step_by(18) {
        grid13_array_reshape[i] = grid_width_13 * w_cnt + grid_width_13 * grid13_array_reshape[i]; //rm-sigmoid
        grid13_array_reshape[i + 1] =
            grid_width_13 * h_cnt + grid_width_13 * grid13_array_reshape[i + 1]; //rm-sigmoid
        grid13_array_reshape[i + 2] = anchor_box_13[0][0] * (grid13_array_reshape[i + 2]).exp();
        grid13_array_reshape[i + 3] = anchor_box_13[0][1] * (grid13_array_reshape[i + 3]).exp();

        grid13_array_reshape[i + 6] =
            grid_width_13 * w_cnt + grid_width_13 * grid13_array_reshape[i + 6]; //rm-sigmoid
        grid13_array_reshape[i + 7] =
            grid_width_13 * h_cnt + grid_width_13 * grid13_array_reshape[i + 7]; //rm-sigmoid
        grid13_array_reshape[i + 8] = anchor_box_13[1][0] * (grid13_array_reshape[i + 8]).exp();
        grid13_array_reshape[i + 9] = anchor_box_13[1][1] * (grid13_array_reshape[i + 9]).exp();

        grid13_array_reshape[i + 12] =
            grid_width_13 * w_cnt + grid_width_13 * grid13_array_reshape[i + 12]; //rm-sigmoid
        grid13_array_reshape[i + 13] =
            grid_width_13 * h_cnt + grid_width_13 * grid13_array_reshape[i + 13]; //rm-sigmoid
        grid13_array_reshape[i + 14] = anchor_box_13[2][0] * (grid13_array_reshape[i + 14]).exp();
        grid13_array_reshape[i + 15] = anchor_box_13[2][1] * (grid13_array_reshape[i + 15]).exp();

        w_cnt += 1.;
        if w_cnt == (grid_13 as f32) {
            w_cnt = 0.;
            h_cnt += 1.;
        }
    }
    w_cnt = 0.;
    h_cnt = 0.;
    for i in (0..grid_26 * grid_26 * 18).step_by(18) {
        grid26_array_reshape[i] = grid_width_26 * w_cnt + grid_width_26 * grid26_array_reshape[i]; //rm-sigmoid
        grid26_array_reshape[i + 1] =
            grid_width_26 * h_cnt + grid_width_26 * grid26_array_reshape[i + 1]; //rm-sigmoid
        grid26_array_reshape[i + 2] = anchor_box_26[0][0] * (grid26_array_reshape[i + 2]).exp();
        grid26_array_reshape[i + 3] = anchor_box_26[0][1] * (grid26_array_reshape[i + 3]).exp();

        grid26_array_reshape[i + 6] =
            grid_width_26 * w_cnt + grid_width_26 * grid26_array_reshape[i + 6]; //rm-sigmoid
        grid26_array_reshape[i + 7] =
            grid_width_26 * h_cnt + grid_width_26 * grid26_array_reshape[i + 7]; //rm-sigmoid
        grid26_array_reshape[i + 8] = anchor_box_26[1][0] * (grid26_array_reshape[i + 8]).exp();
        grid26_array_reshape[i + 9] = anchor_box_26[1][1] * (grid26_array_reshape[i + 9]).exp();

        grid26_array_reshape[i + 12] =
            grid_width_26 * w_cnt + grid_width_26 * grid26_array_reshape[i + 12]; //rm-sigmoid
        grid26_array_reshape[i + 13] =
            grid_width_26 * h_cnt + grid_width_26 * grid26_array_reshape[i + 13]; //rm-sigmoid
        grid26_array_reshape[i + 14] = anchor_box_26[2][0] * (grid26_array_reshape[i + 14]).exp();
        grid26_array_reshape[i + 15] = anchor_box_26[2][1] * (grid26_array_reshape[i + 15]).exp();

        w_cnt += 1.;
        if w_cnt == (grid_26 as f32) {
            w_cnt = 0.;
            h_cnt += 1.;
        }
    }

    // 13*13検出と26*26検出を結合
    // 13*13*255, 26*26*255 >> (13*13+26*26)*255
    let mut grid13and26_array: Vec<f32> = vec![];
    for i in 0..grid_13 * grid_13 * 18 {
        grid13and26_array.push(grid13_array_reshape[i]);
    }
    for i in 0..grid_26 * grid_26 * 18 {
        grid13and26_array.push(grid26_array_reshape[i]);
    }

    let mut grid13and26_array_class: Vec<f32> = vec![];
    for i in 0..grid_13 * grid_13 * anchor_box_num * cls_num {
        grid13and26_array_class.push(grid13_array_class[i]);
    }
    for i in 0..grid_26 * grid_26 * anchor_box_num * cls_num {
        grid13and26_array_class.push(grid26_array_class[i]);
    }

    let mut cnt_cls = 0;
    let mut nms_boxes: Vec<DetectionData> = vec![];
    for i in 0..(grid_13 * grid_13 + grid_26 * grid_26) * anchor_box_num {
        let cx = grid13and26_array[i * (18 / anchor_box_num) + 0];
        let cy = grid13and26_array[i * (18 / anchor_box_num) + 1];
        let cw = grid13and26_array[i * (18 / anchor_box_num) + 2];
        let ch = grid13and26_array[i * (18 / anchor_box_num) + 3];
        let obj_score = grid13and26_array[i * (18 / anchor_box_num) + 4];
        let mut max_class_score = -1.0;
        let mut max_class_id = -1;
        for j in 0..cls_num {
            let t = grid13and26_array_class[cnt_cls + j];
            if max_class_score < t && 0.0 <= t && t <= 1.0 {
                max_class_score = t;
                max_class_id = j as i32;
            }
        }
        if max_class_id == -1 {
            println!("cls index out ouf range cls-conf : {}", i);
            continue;
        }
        cnt_cls += cls_num;
        // let class_score = max_class_score;
        let class_id = max_class_id as u8;
        let nms_box = DetectionData {
            class: class_id,
            x1: cx - cw / 2.,
            y1: cy - ch / 2.,
            x2: cx + cw / 2.,
            y2: cy + ch / 2.,
            confidence: obj_score,
        };
        if (0. <= nms_box.x1 && nms_box.x1 <= 416.)
            && (0. <= nms_box.y1 && nms_box.y1 <= 416.)
            && (0. <= nms_box.x2 && nms_box.x2 <= 416.)
            && (0. <= nms_box.y2 && nms_box.y2 <= 416.)
        {
            nms_boxes.push(nms_box);
        } else {
            error!("nms_box out of range: {:?}", nms_box);
        }
    }
    nms_process(&nms_boxes, cls_num, obj_threshold, nms_threshold)
}

fn iou(a: &DetectionData, b: &DetectionData) -> f32 {
    let mut x1 = a.x1;
    let mut y1 = a.y1;
    let mut xx1 = a.x2;
    let mut yy1 = a.y2;
    let mut x2 = b.x1;
    let mut y2 = b.y1;
    let mut xx2 = b.x2;
    let mut yy2 = b.y2;
    if b.x1 > a.x1 {
        x1 = b.x1;
        x2 = a.x1;
    }
    if b.y1 > a.y1 {
        y1 = b.y1;
        y2 = a.y1;
    }
    if b.x2 < a.x2 {
        xx1 = b.x2;
        xx2 = a.x2;
    }
    if b.y2 < a.y2 {
        yy1 = b.y2;
        yy2 = a.y2;
    }
    if x1 >= xx1 || y1 >= yy1 {
        return 0.0;
    }
    let area1 = (xx1 - x1) * (yy1 - y1);
    let area2 = (xx2 - x2) * (yy2 - y2);
    return area1 as f32 / area2 as f32;
}

fn nms(bb: &[DetectionData], nms_threshold: f32) -> Vec<DetectionData> {
    let mut sorted_bb = bb.to_vec();
    sorted_bb.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

    for ib in 0..sorted_bb.len() {
        let mut it = ib + 1;
        while it < sorted_bb.len() {
            if iou(&sorted_bb[ib], &sorted_bb[it]) > nms_threshold {
                sorted_bb.remove(it);
            } else {
                it += 1;
            }
        }
    }
    sorted_bb
}

fn nms_process(
    bb: &[DetectionData],
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
) -> Vec<DetectionData> {
    let mut cls: Vec<Vec<DetectionData>> = vec![vec![]; cls_num];
    for i in 0..bb.len() {
        if bb[i].confidence > obj_threshold && bb[i].confidence <= 1.0 {
            cls[bb[i].class as usize].push(bb[i]);
        }
    }
    let mut new_box = vec![];
    for i in 0..cls_num {
        if !cls[i].is_empty() {
            let nms_cls = nms(&cls[i], nms_threshold);
            new_box.extend(nms_cls);
        }
    }
    new_box
}

pub fn reverse_transform(
    width: u32,
    height: u32,
    rotate_angle: u32,
    d_result: &[DetectionData],
) -> Vec<DetectionData> {
    let mut result = vec![];

    for d in d_result.iter() {
        let mut new_d = *d;
        (new_d.x1, new_d.y1) = point_reverse_transform(width, height, rotate_angle, d.x1, d.y1);
        (new_d.x2, new_d.y2) = point_reverse_transform(width, height, rotate_angle, d.x2, d.y2);
        result.push(new_d);
    }
    result
}

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
    let nw = f32::max(width as f32 * ratio, 1.);
    let nh = f32::max(height as f32 * ratio, 1.);

    let pad_w = (nw - yolo_input_size).abs() / 2.;
    let pad_h = (nh - yolo_input_size).abs() / 2.;

    ((x - pad_w) / ratio, (y - pad_h) / ratio)
}
