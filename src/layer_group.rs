use anyhow::{bail, Result};


#[derive(Clone, Copy, PartialEq)]
/// 活性化関数の種類を表す列挙型
pub enum Activation {
    Linear,
    Leaky,
}

#[derive(Clone, Copy, PartialEq)]
/// ポストプロセスの種類を表す列挙型
pub enum PostProcess {
    None,
    MaxPool,
    Yolo,
    Upsample,
}

/// レイヤーグループの構造体
pub struct LayerGroup {
    pub input_width: u32,
    pub input_height: u32,
    pub input_ch: u32,
    pub input_fold_ch: u32,
    pub input_size: u32,
    /// 1グループにつき何回モジュールを使い回すか
    /// fold_factor は, その層のチャネル数 / 一度に処理できる最大のチャネル数
    pub input_fold_factor: u32,
    pub acc_size: u32,

    pub output_width: u32,
    pub output_height: u32,
    pub output_ch: u32,
    pub output_fold_ch: u32,
    pub output_size: u32,
    pub output_fold_factor: u32,
    pub pooling_stride: u32,

    pub inputs: Option<Vec<i16>>,
    pub outputs: Option<Vec<i16>>,
    pub weights: Option<Vec<i16>>,
    pub biases: Option<Vec<i16>>,

    pub activate_type: Activation,
    pub post_process_type: PostProcess,
    pub conv_disable: bool,
}

const CH_FOLD_FACTOR: u32 = 4;

impl LayerGroup {
    pub fn new(
        input_w: u32,
        input_h: u32,
        input_ch: u32,
        input_fold_factor: u32,
        output_w: u32,
        output_h: u32,
        output_ch: u32,
        output_fold_factor: u32,
        conv_disable: bool,
        activate_type: Activation,
        post_process_type: PostProcess,
        pooling_stride: u32,
    ) -> Self {
        let input_fold_ch = (input_ch + 3) / 4;
        let output_fold_ch = (output_ch + 3) / 4;
        Self {
            input_width: input_w,
            input_height: input_h,
            input_ch,
            input_fold_ch,
            input_size: input_w * input_h * input_fold_ch * CH_FOLD_FACTOR,
            input_fold_factor,
            output_width: output_w,
            output_height: output_h,
            output_ch,
            output_fold_ch,
            output_size: output_w * output_h * output_fold_ch * CH_FOLD_FACTOR,
            output_fold_factor,
            acc_size: input_w * input_h * output_fold_ch * CH_FOLD_FACTOR,
            conv_disable,
            activate_type,
            post_process_type,
            pooling_stride,
            inputs: None,
            outputs: None,
            weights: None,
            biases: None,
        }
    }
    pub fn get_weights(&self, off: u32, iff: u32) -> Result<&[i16]> {
        match &self.weights {
            Some(w) => {
                let weight_size = 12 * self.input_ch * self.output_ch;
                let data_beg = (weight_size * self.output_fold_factor * iff + 12 * weight_size * off) as usize;
                let data_end = data_beg + weight_size as usize;
                Ok(&w[data_beg..data_end])
            },
            None => bail!("Weight is not set")
        }
    }
    pub fn get_inputs(&self, iff: u32) -> Result<&[i16]> {
        match &self.inputs {
            Some(i) => {
                let data_beg = (self.input_size * iff) as usize;
                let data_end = data_beg + self.input_size as usize;
                Ok(&i[data_beg..data_end])
            },
            None => bail!("Input is not set")
        }
    }
    pub fn get_biases(&self, off: u32) -> Result<&[i16]> {
        match &self.biases {
            Some(b) => {
                let data_beg = (self.output_ch * off) as usize;
                let data_end = data_beg + self.output_ch as usize;
                Ok(&b[data_beg..data_end])
            },
            None => bail!("Bias is not set")
        }
    }
    pub fn set_outputs(&mut self, off: u32, output: Vec<i16>) {
        let data_beg = (self.output_size * off) as usize;
        let data_end = data_beg + self.output_size as usize;
        match &mut self.outputs {
            Some(o) => {
                o.splice(data_beg..data_end, output);
            },
            None => {
                if off == 0 {
                    self.outputs = Some(output);
                }
                else {
                    let mut new_output = vec![0; data_beg];
                    new_output.extend(output);
                    self.outputs = Some(new_output);
                }
            }
        }
    }
}
