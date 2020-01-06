use serde::{Deserialize, Serialize};
use std::io::prelude::*;
use std::fs::File;

//rust binary 二进制文件读写
fn read_a_file(filename:String) -> Vec<u8> 
{
    let mut file = File::open(filename).expect("file not found");

    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("file read_to_end error");
    data
}

fn write_a_file(filename:String, data:Vec<u8>)
{
    let mut buffer = File::create(filename).expect("file create error");
    buffer.write_all(&data).expect("file write error");
}

#[derive(Serialize, Deserialize, Debug)]
struct Person {
    name:   String,
    age:    u8,
    phones: Vec<String>,
}

fn main() {
    let data = r#"
        {
            "name": "John Doe",
            "age": 43,
            "phones": [
                "+44 1234567",
                "+44 2345678"
            ]
        }"#;

    let p: Person = serde_json::from_str(data).unwrap();
    println!("Please call {} at the number {} age {}", p.name, p.phones[0], p.age);


    let p2 = Person {
               name:String::from("wsk"),
               age:12,
               phones:vec![String::from("12111"),String::from("1293878")]};

    let p2_ser = serde_json::to_string(&p2).unwrap();
    println!("{}", p2_ser);

    //存储序列化数据
    let mut p2_ser_vec_char = p2_ser.into_bytes();
    //to_le_bytes() x86和网络字节序默认小端机器;因此就设置长度为小端存储 
    //这里把usize强转为u64也是为了适应32/64位机器，usize长度不固定
    let mut ser_byte_len:Vec<u8> = (p2_ser_vec_char.len() as u64).to_le_bytes().to_vec();
    ser_byte_len.append(&mut p2_ser_vec_char);//最开始8个字节为序列化的长度
    let mut data:Vec<u8> = vec![1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6];//数据
    ser_byte_len.append(&mut data);
    println!("{:?} ", ser_byte_len);//打包好的内存
    write_a_file("test.bin".to_string(), ser_byte_len);//存成二进制文件

    //解压序列化数据
    let rdf = read_a_file("test.bin".to_string());//读二进制文件
    let tes:u64 = unsafe {std::ptr::read(rdf.as_ptr() as *const _) };//将首地址强转为u64，读取序列化头的长度
    let rs = String::from_utf8_lossy(&rdf[8..(tes + 8) as usize]);//将序列化二进制内存转成字符串
    println!("{}",rs);
    let tp: Person = serde_json::from_str(&rs).unwrap();//反序列化
    println!("name {} number {} age {}", tp.name, tp.phones[0], tp.age);
}

