use anyhow::{bail, Context, Result};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use image::DynamicImage;
use v4l::buffer::Type;
use v4l::io::{mmap::Stream, traits::CaptureStream};
use v4l::video::Capture;
use v4l::{Device, FourCC};

use tiny_yolo_v3_zynq_rs::img_proc::draw_bbox;
use tiny_yolo_v3_zynq_rs::yolo::YoloV3Tiny;

fn main() -> Result<()> {
    let cam_device_index = 2;
    let frame_width = 640;
    let frame_height = 480;
    let wdir = "examples/weights";

    // YOLO IP を初期化
    let mut yolo = YoloV3Tiny::new("/slab/hwinfo.json", "yolo", 7, 0.2, 0.1, wdir, wdir)?;

    // YOLOの処理中にもカメラのバッファを更新する必要があるため，マルチスレッドでカメラだけ動かしておく
    // 動かしておかないと (YOLOの実行時間) * (カメラのバッファ数: 3) 秒前の画像になる
    // もしかしたらもっといい方法があるかも？
    let loader = CamImgLoader::new(cam_device_index, frame_width, frame_height);

    // ./out ディレクトリを作成
    std::fs::create_dir_all("./out")?;

    for _ in 0..10 {
        let start = Instant::now();
        let img = loader.receive()?;
        let result = yolo.start(&img, 90)?;

        let end = start.elapsed();
        let t = end.as_secs_f64() * 1000.0;
        println!("Processing time:{:.03}ms, {:.1}FPS", t, 1000. / t);

        let mut rgb_img = img.rotate90().to_rgb8();
        draw_bbox(&mut rgb_img, &result);
        rgb_img.save(format!("./out/out.png"))?;
    }
    Ok(())
}

/// カメラ画像を取得するための構造体
struct CamImgLoader {
    /// スレッドハンドル
    thread_handle: Option<thread::JoinHandle<()>>,
    /// start, stopなどコマンドのsender
    cmd_tx: mpsc::Sender<String>,
    /// カメラ画像のsender
    cam_img_rx: mpsc::Receiver<DynamicImage>,
}

impl CamImgLoader {
    /// コンストラクタ
    fn new(cam_device_index: usize, frame_width: u32, frame_height: u32) -> Self {
        // 変数のcloneとか
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (cam_img_tx, cam_img_rx) = mpsc::channel();

        // スレッドの開始
        let thread_handle = Some(thread::spawn(move || {
            let _ = Self::run_cam_thread(
                cam_device_index,
                cmd_rx,
                cam_img_tx,
                frame_width,
                frame_height,
            );
        }));
        Self {
            thread_handle,
            cmd_tx,
            cam_img_rx,
        }
    }

    /// スレッドの中身
    fn run_cam_thread(
        cam_device_index: usize,
        cmd_rx: mpsc::Receiver<String>,
        cam_img_tx: mpsc::Sender<DynamicImage>,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<()> {
        // カメラデバイスをOpen
        let mut dev = Device::new(cam_device_index)?;

        // カメラのフォーマットを設定
        let mut fmt = dev.format()?;
        fmt.width = frame_width;
        fmt.height = frame_height;
        fmt.fourcc = FourCC::new(b"MJPG");
        dev.set_format(&fmt)?;

        let mut cam_stream = Stream::with_buffers(&mut dev, Type::VideoCapture, 3)?;

        loop {
            let (frame, _meta) = CaptureStream::next(&mut cam_stream)?;
            let img = image::load_from_memory(frame)?;

            // コマンドの待機
            if let Ok(msg) = cmd_rx.try_recv() {
                // stopならスレッド終了
                if msg == "stop" {
                    break;
                } else {
                    cam_img_tx.send(img)?;
                }
            }
            thread::yield_now();
        }
        Ok(())
    }

    /// 画像の取得を開始します。
    pub fn start(&self) -> Result<()> {
        // スレッドが停止していないか？
        if self.thread_handle.is_some() {
            // startコマンドの送信
            self.cmd_tx.send(String::from("start"))?;
        }
        Ok(())
    }

    /// 画像をスレッドから受信します。
    pub fn receive(&self) -> Result<DynamicImage> {
        self.start()?;
        Ok(self.cam_img_rx.recv()?)
    }

    /// スレッドを停止します。
    pub fn stop(&mut self) -> Result<()> {
        // スレッドがすでに停止しているか？
        if self.thread_handle.is_some() {
            // stopコマンドの送信
            self.cmd_tx.send(String::from("stop"))?;

            // スレッドをjoin
            let j = self
                .thread_handle
                .take()
                .context("Can't take thread_handle")?
                .join();
            if let Err(_) = j {
                bail!("Can't join thread");
            }
        }
        Ok(())
    }
}

impl Drop for CamImgLoader {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
