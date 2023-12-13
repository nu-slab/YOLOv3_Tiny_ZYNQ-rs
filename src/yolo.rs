//! YOLOのモデルをコントロールするモジュール

use std::fs::File;
use std::{ffi::OsStr, io::Read, path::Path, vec};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use log::{warn, info};
use tar::Archive;

use xipdriver_rs::{axidma, axis_switch, yolo};

use crate::layer_group::{Activation, LayerGroup, PostProcess};

const ACTIVE_EN: [u32; 8] = [
    0xfffffff3, 0xffffffff, 0xfe7fffff, 0xffffffff, 0xffffffff, 0xffffcfff, 0xffffffff, 0x7fffffff,
];

/// YOLOのモデルをコントロールする構造体
pub struct YoloController {
    /// AxisSwitchのインスタンス0
    sw0: axis_switch::AxisSwitch,
    /// AxisSwitchのインスタンス1
    sw1: axis_switch::AxisSwitch,
    /// AxisSwitchのインスタンス2
    sw2: axis_switch::AxisSwitch,
    /// AxiDmaのインスタンス0
    dma0: axidma::AxiDma,
    /// AxiDmaのインスタンス1
    dma1: axidma::AxiDma,
    /// YOLOアクセラレータのインスタンス
    yolo_acc: yolo::Yolo,
    /// YOLO畳み込み層のインスタンス
    yolo_conv: yolo::Yolo,
    /// YOLO最大プーリング層のインスタンス
    yolo_mp: yolo::Yolo,
    /// YOLO層のインスタンス
    yolo_yolo: yolo::Yolo,
    /// YOLOアップサンプリング層のインスタンス
    yolo_upsamp: yolo::Yolo,
    /// レイヤーグループのベクトル
    pub(crate) layer_groups: Vec<LayerGroup>,
}

impl YoloController {
    /// 新たな `YoloController` のインスタンスを作成します。
    ///
    /// # Args
    /// * `hwinfo_path` - ハードウェア情報のパス
    /// * `yolo_hier` - YOLOの階層名
    ///
    /// # 返り値
    /// * 新たな `YoloController` のインスタンス
    pub fn new(hwinfo_path: &str, yolo_hier: &str) -> Result<Self> {
        // ハードウェア情報の読み込み
        let hw_json = xipdriver_rs::hwinfo::read(hwinfo_path)?;

        // ハードウェア名を取得
        let sw0_name = format!("/{}/{}", yolo_hier, "axis_switch_0");
        let sw1_name = format!("/{}/{}", yolo_hier, "axis_switch_1");
        let sw2_name = format!("/{}/{}", yolo_hier, "axis_switch_2");

        let dma0_name = format!("/{}/{}", yolo_hier, "axi_dma_0");
        let dma1_name = format!("/{}/{}", yolo_hier, "axi_dma_1");

        let yolo_acc_name = format!("/{}/{}", yolo_hier, "yolo_acc_top_0");
        let yolo_conv_name = format!("/{}/{}", yolo_hier, "yolo_conv_top_0");
        let yolo_mp_name = format!("/{}/{}", yolo_hier, "yolo_max_pool_top_0");
        let yolo_yolo_name = format!("/{}/{}", yolo_hier, "yolo_yolo_top_0");
        let yolo_upsamp_name = format!("/{}/{}", yolo_hier, "yolo_upsamp_top_0");

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

        Ok(Self {
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
        })
    }

    /// YOLOの畳み込み層の設定を行います。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
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

    /// YOLOの最大プーリング層の設定を行います。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `add_val` - 追加値
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

    /// YOLOのYOLO層の設定を行います。
    ///
    /// # Args
    /// * `active_en` - アクティブ化関数の有効化フラグ
    /// * `input_h` - 入力の高さ
    /// * `input_w` - 入力の幅
    fn set_yolo_yolo(&self, active_en: u32, input_h: u32, input_w: u32) {
        self.yolo_yolo.set("ACTIVATE_EN", active_en);
        self.yolo_yolo.set("INPUT_H", input_h);
        self.yolo_yolo.set("INPUT_W", input_w);
    }

    /// YOLOのアキュムレータ層の設定を行います。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `bias_en` - バイアスの有効化フラグ
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

    /// Axi4-Stream Switchの設定を行います。
    ///
    /// # Args
    /// * `conv_disable` - 畳み込みの無効化フラグ
    /// * `post_process_type` - ポストプロセスのタイプ
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

    /// 全てのIPをスタートします。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
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

    /// 全てのIPの設定を行います。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `i` - インデックス
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

    /// 畳み込みとアキュムレータIPの設定を行います。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    fn configure_conv_and_acc_ips(&self, grp_idx: usize) {
        self.set_yolo_conv(grp_idx);
        self.set_yolo_acc(grp_idx, false);
        self.set_axis_switch(false, PostProcess::None);
        self.yolo_conv.start();
        self.yolo_acc.start();
    }

    /// 重みを転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `off` - オフセット
    /// * `iff` - インデックス
    ///
    /// # 返り値
    /// * Result。転送に失敗した場合はエラー
    fn transfer_weights(&mut self, grp_idx: usize, off: u32, iff: u32) -> Result<()> {
        // キャッシュは無効なので，Flushはしなくていい (はず)
        let weights = self.layer_groups[grp_idx].get_weights(off, iff)?;
        self.dma0.write(weights)?;
        while !self.dma0.is_mm2s_idle()? {}
        Ok(())
    }

    /// バイアスを転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `off` - オフセット
    ///
    /// # 返り値
    /// * Result。転送に失敗した場合はエラー
    fn transfer_biases(&mut self, grp_idx: usize, off: u32) -> Result<()> {
        let biases = self.layer_groups[grp_idx].get_biases(off)?;
        self.dma1.write(biases)?;
        while !self.dma1.is_mm2s_idle()? {}
        Ok(())
    }

    /// アキュムレータの入力を転送します。
    ///
    /// # Args
    /// * `acc_input_buff` - アキュムレータの入力バッファ
    ///
    /// # 返り値
    /// * Result。転送に失敗した場合はエラー
    fn transfer_acc_input(&mut self, acc_input_buff: &[i16]) -> Result<()> {
        self.dma1.write(acc_input_buff)
    }

    /// アキュムレータの出力を転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    ///
    /// # 返り値
    /// * アキュムレータの出力を含むVec<i16>のResult。転送に失敗した場合はエラー
    fn transfer_acc_output(&mut self, grp_idx: usize) -> Result<Vec<i16>> {
        self.dma0.read(self.layer_groups[grp_idx].acc_size as usize)
    }

    /// 出力を転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    ///
    /// # 返り値
    /// * 出力を含むVec<i16>のResult。転送に失敗した場合はエラー
    fn transfer_output(&mut self, grp_idx: usize) -> Result<Vec<i16>> {
        self.dma0
            .read(self.layer_groups[grp_idx].output_size as usize)
    }

    /// 入力を転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `idx` - インデックス
    ///
    /// # 返り値
    /// * Result。転送に失敗した場合はエラー
    fn transfer_inputs(&mut self, grp_idx: usize, idx: u32) -> Result<()> {
        let inputs = self.layer_groups[grp_idx].get_inputs(idx)?;
        self.dma0.write(inputs)?;
        Ok(())
    }
    /// 最後のチャネルデータを転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `off` - オフセット
    /// * `iff` - インデックス
    /// * `acc_input_buff` - アキュムレータの入力バッファ
    ///
    /// # 返り値
    /// * Result。転送に失敗した場合はエラー
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

    /// サブチャネルデータを転送します。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
    /// * `iff` - インデックス
    /// * `acc_input_buff` - アキュムレータの入力バッファ
    /// * `acc_output_buff` - アキュムレータの出力バッファ
    ///
    /// # 返り値
    /// * Result。転送に失敗した場合はエラー
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

    /// 全てのIPが完了するまで待ちます。
    ///
    /// # Args
    /// * `grp_idx` - レイヤーグループのインデックス
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

    /// アキュムレータIPが完了するまで待ちます。
    fn wait_acc_ip(&self) {
        while !self.yolo_acc.is_done() {}
    }

    /// レイヤーグループの処理を開始します。
    ///
    /// # Args
    /// * `grp_idx` - 処理を開始するレイヤーグループのインデックス
    ///
    /// # 返り値
    /// * Result。処理に失敗した場合はエラー
    pub fn start_layer_processing(&mut self, grp_idx: usize) -> Result<()> {
        for off in 0..self.layer_groups[grp_idx].output_fold_factor {
            let mut acc_output_buff = vec![];
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

                std::mem::swap(&mut acc_input_buff, &mut acc_output_buff);
            }
        }
        Ok(())
    }

    /// 重みデータを読み込みます。
    ///
    /// # Args
    /// * `weights_dir` - 重みデータが格納されているディレクトリへのパス
    ///
    /// # 注意
    /// この関数は各レイヤーグループの重みデータを読み込みます。データは16ビット整数として解釈されます。
    /// ファイルが存在しない場合、そのレイヤーグループの重みは更新されません。
    pub fn _read_weights<S: AsRef<OsStr> + ?Sized>(&mut self, weights_dir: &S) {
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

    /// バイアスデータを読み込みます。
    ///
    /// # Args
    /// * `biases_dir` - バイアスデータが格納されているディレクトリへのパス
    ///
    /// # 注意
    /// この関数は各レイヤーグループのバイアスデータを読み込みます。データは16ビット整数として解釈されます。
    /// ファイルが存在しない場合、そのレイヤーグループのバイアスは更新されません。
    pub fn _read_biases<S: AsRef<OsStr> + ?Sized>(&mut self, biases_dir: &S) {
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
        let file = File::open(path)?;
        let mut archive = Archive::new(GzDecoder::new(file));

        for file in archive.entries()? {
            let mut file = file?;
            let file_path = file.path()?;
            let file_name = file_path
                .file_name()
                .context("file_name error")?
                .to_str()
                .context("to_str error")?
                .to_string();

            // Skip files that start with '._'
            if file_name.starts_with("._") {
                continue;
            }

            let mut buf = vec![];
            file.read_to_end(&mut buf).unwrap();
            let data: Vec<i16> = buf
                .chunks(2)
                .map(|chunk| {
                    let bytes = [chunk[0], chunk[1]];
                    i16::from_le_bytes(bytes)
                })
                .collect();

            if &file_name[..6] == "biases" {
                let gnum: usize = file_name[6..].parse()?;
                info!("Loading bias {}", gnum);
                self.layer_groups[gnum].biases = Some(data);
            } else if &file_name[..7] == "weights" {
                let gnum: usize = file_name[7..].parse()?;
                info!("Loading weight {}", gnum);
                self.layer_groups[gnum].weights = Some(data);
            } else {
                warn!("{} is not biases or weights file", file_name);
            }
        }
        Ok(())
    }

    /// DMAを停止します
    pub fn stop_dmas(&self) {
        self.dma0.stop();
        self.dma1.stop();
    }
}

impl Drop for YoloController {
    // デストラクタ (スレッドを停止)
    fn drop(&mut self) {
        self.stop_dmas();
    }
}
