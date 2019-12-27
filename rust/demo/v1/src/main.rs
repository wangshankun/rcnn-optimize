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
    let mut file = File::open("100.jpg").expect("file not found");
    let mut buffer = Vec::<u8>::new();
    let mut buf_len = 0;
    match  file.read_to_end(&mut buffer)
    {
        //Ok(read_len) => println!("{}", read_len);buf_len = read_len,
        Ok(read_len) => buf_len = read_len,//赋值给外部的buf_len，而read_len本身生命周期只有这行
        Err(..) => {}
    }

    let cp1 = CompressInputImage { 
                  image_id:      "1000_9919".to_string(),
                  channel_id:    "hangzhou_xxx_".to_string(),
                  ts_ms:         1212212,
                  compress_rate: 16,
                  image_format:  0,
                  buf:           buffer,
                  buf_len:       buf_len as u64, //强转 uszie 到 u64
                 };

    println!("cp1 is {:?}", cp1);
}
