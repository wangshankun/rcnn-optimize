use std::fs::File;
use std::io::prelude::*;
use std::collections::HashMap;

extern crate rand;
use rand::{Rng, thread_rng};

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CompressInputImage {
    image_id:      String,
    channel_id:    String,
    ts_ms:         u64,
    compress_rate: i32,
    image_format:  i32,
    buf:           Vec::<u8>,
    buf_len:       u64,
}

fn compress_images(vec_cmp_in: &Vec<CompressInputImage>)
{
//    println!(" {} {}", vec_cmp_in[3].image_id, vec_cmp_in[3].channel_id);

    let mut channel_hit_list:HashMap<&String, Vec<&CompressInputImage>> = HashMap::new();   

    for x in vec_cmp_in
    {   //如果遇到不存在channel id那么就创建一个新的img收集器，把这个img放到收集器中；
        //如果存在就直接放到已经存channel id对应的收集器中
        channel_hit_list.entry(&x.channel_id).or_insert(Vec::<&CompressInputImage>::new()).push(x);
    }
    //println!("{:?}", channel_hit_list);

//    println!("{}", channel_hit_list.len());//三个不同channel id
//    println!("{}", channel_hit_list[&"hangzhou_0192_13".to_string()].len());//属于13 id 的成员有几个
//    println!("{}", channel_hit_list[&"hangzhou_0192_12".to_string()].len());//属于12 id 的成员有几个
//    println!("{}", channel_hit_list[&"hangzhou_0192_11".to_string()].len());//属于11 id 的成员有几个
/*
    for (key, mut val) in channel_hit_list
    {
         println!("channel id :{} have {} members", key, val.len());
         for x in &val
         {
              println!("{}",x.ts_ms);
         }

         println!("根据时间戳逆序排列");
         val.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));

         for x in &val
         {
              println!("{}",x.ts_ms);
         }
    }
*/
    for (key, mut val) in channel_hit_list
    {
         println!("channel id :{} have {} members", key, val.len());
         //让成员按照时间戳顺序从小到大排列
         val.sort_by(|a, b| a.ts_ms.cmp(&b.ts_ms));
         for x in &val
         {
              println!("{}",x.ts_ms);
         }
    }

}

fn main()
{
    let mut vec_cmp_in :Vec<CompressInputImage> = Vec::new();
    for idx_img in 100..109
    {
        //println!("{} {}", file!(), line!()); //行号，文件文件名
        let img_name    = idx_img.to_string() + ".jpg";//100.jpg  .. 108.jpg
        let mut file    = File::open(img_name).expect("file not found");
        let mut buffer  = Vec::<u8>::new();
        let mut buf_len = 0;
        match  file.read_to_end(&mut buffer)
        {
            Ok(read_len) => buf_len = read_len,//赋值给外部的buf_len，而read_len本身生命周期只有这行
            Err(..) => println!("file read_to_end error")
        }

        let rand_inx = thread_rng().gen_range(1, 4);//随机channel id给例子中使用
        let cp1 = CompressInputImage { 
                      image_id:      "1000_9919".to_string() + &idx_img.to_string(),
                      channel_id:    "hangzhou_0192_1".to_string() + &rand_inx.to_string(),
                      ts_ms:         1212212 + idx_img,
                      compress_rate: 16,
                      image_format:  0,
                      buf:           buffer,
                      buf_len:       buf_len as u64, //强转 uszie 到 u64
                     };

        //println!("cp1 is {:?}", cp1);
        vec_cmp_in.push(cp1);
        //println!("Vector length: {}", vec_cmp_in.len());
    }
    //println!(" {} {}", vec_cmp_in[3].image_id, vec_cmp_in[3].channel_id);

   compress_images(&vec_cmp_in);
}
