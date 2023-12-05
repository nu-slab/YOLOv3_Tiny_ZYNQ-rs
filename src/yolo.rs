use std::{ffi::OsStr, io::Read, path::Path, vec};

use anyhow::{anyhow, bail, Context, Result};
use image::DynamicImage;
use log::info;

use crate::{
    img_proc,
    layer_group::{Activation, LayerGroup, PostProcess},
    utils::{self, DetectionData},
};
use xipdriver_rs::{axidma, axis_switch, json_as_map, json_as_str, yolo};

pub fn match_hw(hw_json: &serde_json::Value, hier_name: &str, hw_name: &str) -> Result<String> {
    let hw_object = json_as_map!(hw_json);
    let full_name = format!("/{}/{}", hier_name, hw_name);
    for k in hw_object.keys() {
        if let Some(_) = k.find(hier_name) {
            if json_as_str!(hw_object[k]["fullname"]) == full_name {
                return Ok(k.clone());
            }
        }
    }
    Err(anyhow!("hw object not found: {}, {}", hier_name, hw_name))
}

const ACTIVE_EN: [u32; 8] = [
    0xfffffff3, 0xffffffff, 0xfe7fffff, 0xffffffff, 0xffffffff, 0xffffcfff, 0xffffffff, 0x7fffffff,
];

/// YOLOのモデルをコントロールする構造体
pub struct YoloV3Tiny {
    sw0: axis_switch::AxisSwitch,
    sw1: axis_switch::AxisSwitch,
    sw2: axis_switch::AxisSwitch,
    dma0: axidma::AxiDma,
    dma1: axidma::AxiDma,
    yolo_acc: yolo::Yolo,
    yolo_conv: yolo::Yolo,
    yolo_mp: yolo::Yolo,
    yolo_yolo: yolo::Yolo,
    yolo_upsamp: yolo::Yolo,
    layer_groups: Vec<LayerGroup>,
    cls_num: usize,
    obj_threshold: f32,
    nms_threshold: f32,
}

impl YoloV3Tiny {
    /// コンストラクタ
    pub fn new<S: AsRef<OsStr> + ?Sized>(
        hwinfo_path: &str,
        yolo_hier: &str,
        cls_num: usize,
        obj_threshold: f32,
        nms_threshold: f32,
        weights_dir: &S,
        biases_dir: &S,
    ) -> Result<Self> {
        // ハードウェア情報の読み込み
        let hw_json = xipdriver_rs::hwinfo::read(hwinfo_path)?;

        // ハードウェア名を取得
        let sw0_name = match_hw(&hw_json, yolo_hier, "axis_switch_0")?;
        let sw1_name = match_hw(&hw_json, yolo_hier, "axis_switch_1")?;
        let sw2_name = match_hw(&hw_json, yolo_hier, "axis_switch_2")?;

        let dma0_name = match_hw(&hw_json, yolo_hier, "axi_dma_0")?;
        let dma1_name = match_hw(&hw_json, yolo_hier, "axi_dma_1")?;

        let yolo_acc_name = match_hw(&hw_json, yolo_hier, "yolo_acc_top_0")?;
        let yolo_conv_name = match_hw(&hw_json, yolo_hier, "yolo_conv_top_0")?;
        let yolo_mp_name = match_hw(&hw_json, yolo_hier, "yolo_max_pool_top_0")?;
        let yolo_yolo_name = match_hw(&hw_json, yolo_hier, "yolo_yolo_top_0")?;
        let yolo_upsamp_name = match_hw(&hw_json, yolo_hier, "yolo_upsamp_top_0")?;

        info!("sw0_name: {}", sw0_name);
        info!("sw1_name: {}", sw1_name);
        info!("sw2_name: {}", sw2_name);
        info!("dma0_name: {}", dma0_name);
        info!("dma1_name: {}", dma1_name);
        info!("yolo_acc_name: {}", yolo_acc_name);
        info!("yolo_conv_name: {}", yolo_conv_name);
        info!("yolo_mp_name: {}", yolo_mp_name);
        info!("yolo_yolo_name: {}", yolo_yolo_name);
        info!("yolo_upsamp_name: {}", yolo_upsamp_name);

        // ハードウェアの構造体を初期化
        let sw0 = axis_switch::AxisSwitch::new(&hw_json[sw0_name])?;
        let sw1 = axis_switch::AxisSwitch::new(&hw_json[sw1_name])?;
        let sw2 = axis_switch::AxisSwitch::new(&hw_json[sw2_name])?;

        let mut dma0 = axidma::AxiDma::new(&hw_json[dma0_name])?;
        let mut dma1 = axidma::AxiDma::new(&hw_json[dma1_name])?;

        let yolo_acc = yolo::Yolo::new(&hw_json[yolo_acc_name])?;
        let yolo_conv = yolo::Yolo::new(&hw_json[yolo_conv_name])?;
        let yolo_mp = yolo::Yolo::new(&hw_json[yolo_mp_name])?;
        let yolo_yolo = yolo::Yolo::new(&hw_json[yolo_yolo_name])?;
        let yolo_upsamp = yolo::Yolo::new(&hw_json[yolo_upsamp_name])?;

        dma0.start();
        dma1.start();

        let mut s = Self {
            sw0,
            sw1,
            sw2,
            dma0,
            dma1,
            yolo_acc,
            yolo_conv,
            yolo_mp,
            yolo_yolo,
            yolo_upsamp,
            layer_groups: vec![],
            cls_num,
            obj_threshold,
            nms_threshold,
        };
        s.init(weights_dir, biases_dir);

        Ok(s)
    }

    fn set_yolo_conv(&self, grp_idx: usize) {
        let l = &self.layer_groups[grp_idx];
        self.set_yolo_conv_internal(
            l.output_ch,
            l.input_ch,
            l.output_fold_ch,
            l.input_fold_ch,
            l.input_height + 2,
            l.input_width + 2,
            l.input_height + 2,
            3,
        );
    }

    fn set_yolo_conv_internal(
        &self,
        output_ch: u32,
        input_ch: u32,
        fold_output_ch: u32,
        fold_input_ch: u32,
        input_h: u32,
        input_w: u32,
        real_input_h: u32,
        fold_win_area: u32,
    ) {
        self.yolo_conv.set("OUTPUT_CH", output_ch);
        self.yolo_conv.set("INPUT_CH", input_ch);
        self.yolo_conv.set("FOLD_OUTPUT_CH", fold_output_ch);
        self.yolo_conv.set("FOLD_INPUT_CH", fold_input_ch);
        self.yolo_conv.set("INPUT_H", input_h);
        self.yolo_conv.set("INPUT_W", input_w);
        self.yolo_conv.set("REAL_INPUT_H", real_input_h);
        self.yolo_conv.set("FOLD_WIN_AREA", fold_win_area);
    }

    fn set_yolo_max_pool(&self, grp_idx: usize, add_val: u32) {
        let l = &self.layer_groups[grp_idx];
        self.set_yolo_max_pool_internal(
            l.output_height + add_val,
            l.output_width + add_val,
            l.input_height,
            l.input_width,
            l.output_fold_ch,
            l.pooling_stride,
        );
    }

    fn set_yolo_max_pool_internal(
        &self,
        output_h: u32,
        output_w: u32,
        input_h: u32,
        input_w: u32,
        input_fold_ch: u32,
        stride: u32,
    ) {
        self.yolo_mp.set("OUTPUT_H", output_h);
        self.yolo_mp.set("OUTPUT_W", output_w);
        self.yolo_mp.set("INPUT_H", input_h);
        self.yolo_mp.set("INPUT_W", input_w);
        self.yolo_mp.set("INPUT_FOLD_CH", input_fold_ch);
        self.yolo_mp.set("STRIDE", stride);
    }

    fn set_yolo_yolo(&self, active_en: u32, input_h: u32, input_w: u32) {
        self.yolo_yolo.set("ACTIVATE_EN", active_en);
        self.yolo_yolo.set("INPUT_H", input_h);
        self.yolo_yolo.set("INPUT_W", input_w);
    }
    fn set_yolo_acc(&self, grp_idx: usize, bias_en: bool) {
        let l = &self.layer_groups[grp_idx];

        let leaky_num = if bias_en {
            l.activate_type as u32
        } else {
            Activation::Linear as u32
        };
        let bias_en = if bias_en { 1 } else { 0 };
        self.set_yolo_acc_internal(
            l.input_height,
            l.input_width,
            l.output_fold_ch,
            leaky_num,
            bias_en,
        );
    }

    fn set_yolo_acc_internal(
        &self,
        input_h: u32,
        input_w: u32,
        fold_input_ch: u32,
        leaky: u32,
        bias_en: u32,
    ) {
        self.yolo_acc.set("INPUT_H", input_h);
        self.yolo_acc.set("INPUT_W", input_w);
        self.yolo_acc.set("FOLD_INPUT_CH", fold_input_ch);
        self.yolo_acc.set("LEAKY", leaky);
        self.yolo_acc.set("BIAS_EN", bias_en);
    }

    fn set_axis_switch(&self, conv_disable: bool, post_process_type: PostProcess) {
        let conv_disable_bool = if conv_disable { 1 } else { 0 };
        let post_process_u8 = post_process_type as u8;
        self.set_axis_switch_internal(
            0,
            conv_disable_bool,
            conv_disable_bool,
            post_process_u8,
            post_process_u8,
            0,
        );
    }

    fn set_axis_switch_internal(
        &self,
        switch_0_s: u8,
        switch_0_m: u8,
        switch_1_s: u8,
        switch_1_m: u8,
        switch_2_s: u8,
        switch_2_m: u8,
    ) {
        self.sw0.reg_update_disable();
        self.sw1.reg_update_disable();
        self.sw2.reg_update_disable();

        self.sw0.disable_all_mi_ports();
        self.sw1.disable_all_mi_ports();
        self.sw2.disable_all_mi_ports();

        self.sw0.enable_mi_port(switch_0_m, switch_0_s);
        self.sw1.enable_mi_port(switch_1_m, switch_1_s);
        self.sw2.enable_mi_port(switch_2_m, switch_2_s);

        self.sw0.reg_update_enable();
        self.sw1.reg_update_enable();
        self.sw2.reg_update_enable();
    }

    fn start_all_ips(&self, grp_idx: usize) {
        let l = &self.layer_groups[grp_idx];
        // IPの動作をスタートさせる (まだデータは送ってないので処理はしてない)
        if !l.conv_disable {
            self.yolo_conv.start();
            self.yolo_acc.start();
        }
        if l.post_process_type == PostProcess::MaxPool {
            self.yolo_mp.start();
        }
        if l.post_process_type == PostProcess::Yolo {
            self.yolo_yolo.start();
        }
        if l.post_process_type == PostProcess::Upsample {
            self.yolo_upsamp.start();
        }
    }

    fn configure_all_ips(&self, grp_idx: usize, i: u32) {
        let l = &self.layer_groups[grp_idx];
        if !l.conv_disable {
            self.set_yolo_conv(grp_idx);
            self.set_yolo_acc(grp_idx, true);
        }
        if l.post_process_type == PostProcess::MaxPool {
            if l.pooling_stride == 2 {
                self.set_yolo_max_pool(grp_idx, 0);
            } else {
                self.set_yolo_max_pool(grp_idx, 1);
            }
        }
        if l.post_process_type == PostProcess::Yolo {
            self.set_yolo_yolo(ACTIVE_EN[i as usize], l.input_height, l.input_width);
        }
        self.set_axis_switch(l.conv_disable, l.post_process_type);
        self.start_all_ips(grp_idx);
    }

    fn configure_conv_and_acc_ips(&self, grp_idx: usize) {
        self.set_yolo_conv(grp_idx);
        self.set_yolo_acc(grp_idx, false);
        self.set_axis_switch(false, PostProcess::None);
        self.yolo_conv.start();
        self.yolo_acc.start();
    }

    fn transfer_weights(&mut self, grp_idx: usize, off: u32, iff: u32) -> Result<()> {
        // キャッシュは無効なので，Flushはしなくていい (はず)
        let weights = self.layer_groups[grp_idx].get_weights(off, iff)?;
        self.dma0.write(weights)?;
        while !self.dma0.is_mm2s_idle()? {}
        Ok(())
    }

    fn transfer_biases(&mut self, grp_idx: usize, off: u32) -> Result<()> {
        let biases = self.layer_groups[grp_idx].get_biases(off)?;
        self.dma1.write(biases)?;
        while !self.dma1.is_mm2s_idle()? {}
        Ok(())
    }

    fn transfer_acc_input(&mut self, acc_input_buff: &[i16]) -> Result<()> {
        self.dma1.write(acc_input_buff)
    }

    fn transfer_acc_output(&mut self, grp_idx: usize) -> Result<Vec<i16>> {
        self.dma0
            .read(self.layer_groups[grp_idx].acc_size as usize)
    }

    fn transfer_output(&mut self, grp_idx: usize) -> Result<Vec<i16>> {
        self.dma0
            .read(self.layer_groups[grp_idx].output_size as usize)
    }

    fn transfer_inputs(&mut self, grp_idx: usize, idx: u32) -> Result<()> {
        let inputs = self.layer_groups[grp_idx].get_inputs(idx)?;
        self.dma0.write(inputs)?;
        Ok(())
    }

    fn transfer_last_channel_data(
        &mut self,
        grp_idx: usize,
        off: u32,
        iff: u32,
        acc_input_buff: &[i16],
    ) -> Result<()> {
        let l = &self.layer_groups[grp_idx];
        if !l.conv_disable {
            // 畳み込み処理がある層のときは,  biasを送ってから入力値を送る
            self.transfer_biases(grp_idx, off)?;

            self.transfer_acc_input(acc_input_buff)?;
            self.transfer_inputs(grp_idx, iff)?;
        } else {
            self.transfer_inputs(grp_idx, off)?;
        }
        let output = self.transfer_output(grp_idx)?;
        self.layer_groups[grp_idx].set_outputs(off, output);

        self.wait_ips(grp_idx);
        Ok(())
    }

    fn transfer_subchannel_data(
        &mut self,
        grp_idx: usize,
        iff: u32,
        acc_input_buff: &[i16],
        acc_output_buff: &mut Vec<i16>,
    ) -> Result<()> {
        self.transfer_inputs(grp_idx, iff)?;
        self.transfer_acc_input(acc_input_buff)?;
        *acc_output_buff = self.transfer_acc_output(grp_idx)?;

        self.wait_acc_ip();
        Ok(())
    }

    fn wait_ips(&self, grp_idx: usize) {
        let l = &self.layer_groups[grp_idx];
        if l.post_process_type == PostProcess::None {
            while !self.yolo_acc.is_done() {}
        }
        if l.post_process_type == PostProcess::MaxPool {
            while !self.yolo_mp.is_done() {}
        }
        if l.post_process_type == PostProcess::Yolo {
            while !self.yolo_yolo.is_done() {}
        }
        if l.post_process_type == PostProcess::Upsample {
            while !self.yolo_upsamp.is_done() {}
        }
    }

    fn wait_acc_ip(&self) {
        while !self.yolo_acc.is_done() {}
    }

    pub fn forward_layer_group(&mut self, grp_idx: usize) -> Result<()> {
        for off in 0..self.layer_groups[grp_idx].output_fold_factor {
            let mut acc_output_buff = vec![0i16; self.layer_groups[grp_idx].acc_size as usize];
            let mut acc_input_buff = vec![0i16; self.layer_groups[grp_idx].acc_size as usize];
            // 最大32チャネルのサブチャネルを処理する
            for iff in 0..self.layer_groups[grp_idx].input_fold_factor {
                // 最後のチャネルか？
                let is_last_input_ch = iff == self.layer_groups[grp_idx].input_fold_factor - 1;

                if is_last_input_ch {
                    // 最後のチャネルならば，IPたちに設定値を送信してスタート
                    self.configure_all_ips(grp_idx, off);
                } else {
                    // 最後のチャネルではなければ，畳み込みだけを実行
                    self.configure_conv_and_acc_ips(grp_idx);
                }

                // 重みパラメータをDMAでFPGA (PL) に転送する
                if !self.layer_groups[grp_idx].conv_disable {
                    self.transfer_weights(grp_idx, off, iff)?;
                }

                // データの送受信
                if is_last_input_ch {
                    self.transfer_last_channel_data(grp_idx, off, iff, &acc_input_buff)?;
                } else {
                    self.transfer_subchannel_data(
                        grp_idx,
                        iff,
                        &acc_input_buff,
                        &mut acc_output_buff,
                    )?;
                }

                let mid_ptr = acc_input_buff;
                acc_input_buff = acc_output_buff;
                acc_output_buff = mid_ptr;
            }
        }
        Ok(())
    }

    #[rustfmt::skip]
    pub fn init<S: AsRef<OsStr> + ?Sized>(&mut self, weights_dir: &S, biases_dir: &S) {
        self.layer_groups.push(LayerGroup::new(416, 416,  3,  1, 208, 208, 16,  1, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.layer_groups.push(LayerGroup::new(208, 208, 16,  1, 104, 104, 32,  1, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.layer_groups.push(LayerGroup::new(104, 104, 32,  1,  52,  52, 32,  2, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.layer_groups.push(LayerGroup::new( 52,  52, 32,  2,  26,  26, 32,  4, false,  Activation::Leaky,  PostProcess::MaxPool, 2));
        self.layer_groups.push(LayerGroup::new( 26,  26, 32,  4,  26,  26, 32,  8, false,  Activation::Leaky,     PostProcess::None, 2));
        self.layer_groups.push(LayerGroup::new( 26,  26, 32,  1,  13,  13, 32,  8,  true, Activation::Linear,  PostProcess::MaxPool, 2));
        self.layer_groups.push(LayerGroup::new( 13,  13, 32,  8,  13,  13, 32, 16, false,  Activation::Leaky,  PostProcess::MaxPool, 1));
        self.layer_groups.push(LayerGroup::new( 13,  13, 32, 16,  13,  13, 32, 32, false,  Activation::Leaky,     PostProcess::None, 2));
        self.layer_groups.push(LayerGroup::new( 13,  13, 32, 32,  13,  13, 32,  8, false,  Activation::Leaky,     PostProcess::None, 2));
        self.layer_groups.push(LayerGroup::new( 13,  13, 32,  8,  13,  13, 32, 16, false,  Activation::Leaky,     PostProcess::None, 2));
        self.layer_groups.push(LayerGroup::new( 13,  13, 32, 16,  13,  13, 32,  8, false, Activation::Linear,     PostProcess::Yolo, 2));
        self.layer_groups.push(LayerGroup::new( 13,  13, 32,  8,  26,  26, 32,  4, false,  Activation::Leaky, PostProcess::Upsample, 2));
        self.layer_groups.push(LayerGroup::new( 26,  26, 32, 12,  26,  26, 32,  8, false,  Activation::Leaky,     PostProcess::None, 2));
        self.layer_groups.push(LayerGroup::new( 26,  26, 32,  8,  26,  26, 32,  8, false, Activation::Linear,     PostProcess::Yolo, 2));

        self.read_weights(weights_dir);
        self.read_biases(biases_dir);
    }

    pub fn read_weights<S: AsRef<OsStr> + ?Sized>(&mut self, weights_dir: &S) {
        for (i, l) in self.layer_groups.iter_mut().enumerate() {
            let path = Path::new(weights_dir).join(format!("weights{}", i));
            if let Ok(mut file) = std::fs::File::open(path) {
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).unwrap();
                l.weights = Some(
                    buf.chunks(2)
                        .map(|chunk| {
                            let bytes = [chunk[0], chunk[1]];
                            i16::from_le_bytes(bytes)
                        })
                        .collect(),
                );
            }
        }
    }
    pub fn read_biases<S: AsRef<OsStr> + ?Sized>(&mut self, biases_dir: &S) {
        for (i, l) in self.layer_groups.iter_mut().enumerate() {
            let path = Path::new(biases_dir).join(format!("biases{}", i));
            if let Ok(mut file) = std::fs::File::open(path) {
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).unwrap();
                l.biases = Some(
                    buf.chunks(2)
                        .map(|chunk| {
                            let bytes = [chunk[0], chunk[1]];
                            i16::from_le_bytes(bytes)
                        })
                        .collect(),
                );
            }
        }
    }

    pub fn start_processing(&mut self, input_data: &[i16]) -> Result<(Vec<i16>, Vec<i16>)> {
        self.layer_groups[0].inputs = Some(Vec::from(input_data));

        for grp_idx in 0..=13 {
            self.forward_layer_group(grp_idx)?;

            if grp_idx == 4 || grp_idx == 8 {
                // あとで使うため，cloneする
                self.layer_groups[grp_idx + 1].inputs = self.layer_groups[grp_idx].outputs.clone();
            } else if grp_idx == 10 {
                // レイヤ11の入力はレイヤ8
                self.layer_groups[11].inputs = self.layer_groups[8].outputs.take();
            } else if grp_idx != 13 {
                // あとで使わないものはmoveして高速化
                self.layer_groups[grp_idx + 1].inputs = self.layer_groups[grp_idx].outputs.take();
            }

            if grp_idx == 11 {
                // レイヤ12の入力はレイヤ11とレイヤ4をconcatしたもの
                // レイヤ11のデータはすでに上でmoveしているので，レイヤ4のデータを結合してあげる
                let output4 = self.layer_groups[4]
                    .outputs
                    .take()
                    .context("layer_groups[4].outputs not set")?;

                match &mut self.layer_groups[12].inputs {
                    Some(inputs) => inputs.extend(output4),
                    None => {
                        bail!("layer_groups[12].inputs not set");
                    }
                }
            }
        }

        // CNNの結果たち
        let output10 = self.layer_groups[10]
            .outputs
            .take()
            .context("layer_groups[10].inputs not set")?;
        let output13 = self.layer_groups[13]
            .outputs
            .take()
            .context("layer_groups[13].inputs not set")?;

        Ok((output10, output13))
    }

    pub fn start(&mut self, img: DynamicImage, rotate_angle: u32) -> Result<Vec<DetectionData>> {
        let img_size = self.layer_groups[0].input_width;
        let input_data = img_proc::letterbox(img, img_size, rotate_angle);

        let (yolo_out_0, yolo_out_1) = self.start_processing(&input_data)?;

        Ok(utils::post_process(
            &yolo_out_0,
            &yolo_out_1,
            self.cls_num,
            self.obj_threshold,
            self.nms_threshold,
        ))
    }

    pub fn stop_dmas(&self) {
        self.dma0.stop();
        self.dma1.stop();
    }
}

impl Drop for YoloV3Tiny {
    // デストラクタ (スレッドを停止)
    fn drop(&mut self) {
        self.stop_dmas();
    }
}
