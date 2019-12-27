use std::fs::File;
use std::io::prelude::*;
#[derive(Debug)]
struct CompressInputImage {
    image_id:      String,
    channel_id:    String,
    ts_ms:         u64,
    compress_rate: i32,
    image_format:  i32,
    buf:           Vec::<u8>,
    buf_len:       u64,
}

fn main()
{
    let mut vec_cmp_in :Vec<CompressInputImage> = Vec::new();
    for idx_img in 100..109
    {
        println!("{} {}", file!(), line!()); //行号，文件文件名
        let img_name    = idx_img.to_string() + ".jpg";//100.jpg  .. 108.jpg
        let mut file    = File::open(img_name).expect("file not found");
        let mut buffer  = Vec::<u8>::new();
        let mut buf_len = 0;
        match  file.read_to_end(&mut buffer)
        {
            Ok(read_len) => buf_len = read_len,//赋值给外部的buf_len，而read_len本身生命周期只有这行
            Err(..) => println!("file read_to_end error")
        }

        let cp1 = CompressInputImage { 
                      image_id:      "1000_9919".to_string() + &idx_img.to_string(),
                      channel_id:    "hangzhou_0192_1".to_string(),
                      ts_ms:         1212212 + idx_img,
                      compress_rate: 16,
                      image_format:  0,
                      buf:           buffer,
                      buf_len:       buf_len as u64, //强转 uszie 到 u64
                     };

        //println!("cp1 is {:?}", cp1);
        vec_cmp_in.push(cp1);
        println!("Vector length: {}", vec_cmp_in.len());
    }
    println!(" {}", vec_cmp_in[3].image_id);
}
