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

fn main() {
    let cp1 = CompressInputImage { 
                  image_id:      "1000_9919".to_string(),
                  channel_id:    "hangzhou_xxx_".to_string(),
                  ts_ms:         1212212,
                  compress_rate: 16,
                  image_format:  0,
                  buf:           Vec::<u8>::new(),
                  buf_len:       0,
                 };

    println!("cp1 is {:?}", cp1);

    let mut file = File::open("100.jpg").expect("file not found");
    let mut buffer = Vec::<u8>::new();
    //let ret   = file.read_to_end(&mut buffer);
    //println!("{:?}", ret);
    match  file.read_to_end(&mut buffer)
    {
        Ok(n) => println!("{}", n),
        Err(..) => {}
    }
   // println!("{:?}", buffer);
}
