#![feature(core_intrinsics)]
#![feature(vec_into_raw_parts)]

use std::ffi::{CStr, CString};
use std::slice;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::io::prelude::*;
use std::fs::File;

#[allow(dead_code)]
#[allow(unused_unsafe)]
fn read_a_file(filename:String) -> Vec<u8>
{
    let mut file = File::open(filename).expect("file not found");

    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("file read_to_end error");
    data
}

#[allow(dead_code)]
#[allow(unused_unsafe)]
fn write_a_file(filename:String, data:&Vec<u8>)
{
    let mut buffer = File::create(filename).expect("file create error");
    buffer.write_all(data).expect("file write error");
}

//在["101","102","103"]中;    找["101","103","108"] 对应的位置下标[0,2]
fn hit_index_vec(a:&Vec<&str>, b:&Vec<&str>) -> Vec<usize>
{
    let mut hit_idx:Vec<usize> = vec![];
    for x in b
    {
        let _ret = match (a).iter().position(|&r| &r == x)
        {
            Some(i) => hit_idx.push(i),
            None => {},//如果请求的id不在，那么忽略
        };
    }
    hit_idx
}

#[allow(dead_code)]
#[allow(unused_unsafe)]
fn print_type_of<T>(_: &T)
{
    println!("{}", unsafe { std::intrinsics::type_name::<T>() });
}

#[repr(C)]
#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CompressInputImage {
    pub image_id:      *const i8,
    pub channel_id:    *const i8,
    pub ts_ms:         u64,
    pub buf:           *const u8,
    pub buf_len:       u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct CompressOutputData {
    pub channel_ids:   *const i8,
    pub image_ids:     *const i8,
    pub ts_arrays:     *const i8,
    pub offsets:       *const i8,
    pub version:       i32,
    pub compress_rate: i32,
    pub image_format:  i32,
    pub buf:           *const u8,
    pub buf_len:       u64,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct SerializeCOData {
    pub channel_ids:   String,
    pub image_ids:     String,
    pub ts_arrays:     String,
    pub offsets:       String,
    pub version:       i32,
    pub compress_rate: i32,
    pub image_format:  i32,
    pub buf:           u64,
    pub buf_len:       u64,
}

#[no_mangle]
pub unsafe extern "C" fn compress_images(cimgs:*mut CompressInputImage, len:usize, compress_rate:i32, image_format:i32, ret_num:*mut usize) 
-> *const CompressOutputData 
{

    let cm_img_array: &[CompressInputImage] = slice::from_raw_parts(cimgs, len as usize);

    let mut channel_hit_list:HashMap<String, Vec<&CompressInputImage>> = HashMap::new();

    for x in cm_img_array
    { 
        let c_id = CStr::from_ptr(x.channel_id).to_str().unwrap().to_owned();
        channel_hit_list.entry(c_id).or_insert(Vec::<&CompressInputImage>::new()).push(x);
    }

    let mut ret_vec:Vec<CompressOutputData> = vec![];
    //按照 channel_id 为一组进行打包逻辑
    for (key, mut val) in channel_hit_list
    {    //根据时间戳从小到大排序
         val.sort_by(|a, b| a.ts_ms.cmp(&b.ts_ms));

         let ch_ids                 = key;
         let mut im_vec:Vec<String> = vec![];
         let mut ts_vec:Vec<u64>    = vec![];
         let mut of_vec:Vec<u64>    = vec![];
         let mut data:Vec<u8>       = vec![];
         let mut ser_data:Vec<u8>   = vec![];
         let mut offset             = 0;

         for x in val
         {
              let a = CStr::from_ptr(x.image_id).to_str().unwrap().to_owned();//*u8转字符串
              im_vec.push(a);
              ts_vec.push(x.ts_ms);
              offset = offset + x.buf_len;
              of_vec.push(offset);

              let img: Vec<u8> = Vec::from_raw_parts(x.buf as *mut _, x.buf_len as usize, x.buf_len as usize);
              data.extend(img);
         }
         let im_ids = im_vec.join(";");
         let ts_ids_v:Vec<_> = ts_vec.iter().map(ToString::to_string).collect();
         let ts_ids = ts_ids_v.join(";");
         let of_ids_v:Vec<_> = of_vec.iter().map(ToString::to_string).collect();
         let of_ids = of_ids_v.join(";");

        //序列化结构体头
        let cod = SerializeCOData
                  {
                      channel_ids:ch_ids.clone(),
                      image_ids:im_ids.clone(),
                      ts_arrays:ts_ids.clone(),
                      offsets:of_ids.clone(),
                      version:0,
                      compress_rate:compress_rate,
                      image_format:image_format,
                      buf:0,
                      buf_len:offset,
                  };  
        let ser_cod = serde_json::to_string(&cod).unwrap();
        //println!("{}", ser_cod);
        let mut ser_vec_char = ser_cod.into_bytes();
        //to_le_bytes() x86和网络字节序默认小端机器;因此就设置长度为小端存储
        //这里把usize强转为i64也是为了适应32/64位机器，usize长度不固定/java没有unsigned long选用long
        let mut ser_head_len:Vec<u8> = (ser_vec_char.len() as i64).to_le_bytes().to_vec();
        ser_data.append(&mut ser_head_len);//最开始8个字节为序列化的长度
        ser_data.append(&mut ser_vec_char);//接下来是序列化包头
        ser_data.append(&mut data);//最后是压缩的数据
        //println!("{}",ser_data.len());
        //将包回传
        let ret = CompressOutputData
                  {
                      channel_ids:CString::new(ch_ids).unwrap().into_raw(),
                      image_ids:CString::new(im_ids).unwrap().into_raw(),
                      ts_arrays:CString::new(ts_ids).unwrap().into_raw(),
                      offsets:CString::new(of_ids).unwrap().into_raw(),
                      version:0,
                      compress_rate:compress_rate,
                      image_format:image_format,
                      buf:ser_data.as_ptr(),
                      buf_len:offset,
                  };
        //println!("{:?}",ret);
        //write_a_file("test_rust.bin".to_string(), &ser_data);//测试:存成二进制文件
        std::mem::forget(ser_data);
        ret_vec.push(ret);
    }

   *ret_num = ret_vec.len();
   let(ptr, _len, _cap) = ret_vec.into_raw_parts();
   ptr
}


#[no_mangle]
pub unsafe extern "C" fn decompress_images(buf:*mut u8, len:i32, hit_ids:*const i8, ret_num:*mut usize)
-> *const CompressInputImage
{
    //包头反序列化
    let buf_v: Vec<u8> = Vec::from_raw_parts(buf, len as usize, len as usize);
    let pkg_head_len:u64 = std::ptr::read(buf_v.as_ptr() as *const _);
    let pkg_head = String::from_utf8_lossy(&buf_v[8..(pkg_head_len + 8) as usize]);//将序列化二进制内存转成字符串

    let dep_info: SerializeCOData = serde_json::from_str(&pkg_head).unwrap();//反序列化
    println!("{:?}",dep_info);

    let hit_img_ids = CStr::from_ptr(hit_ids).to_str().unwrap().to_owned();
    let req_ids_vec: Vec<&str> = hit_img_ids.split(";").collect();
    let src_ids_vec: Vec<&str> = dep_info.image_ids.split(";").collect();
    let hit_vec = hit_index_vec(&src_ids_vec, &req_ids_vec);

    let ts_vec: Vec<u64> = dep_info.ts_arrays
          .split(';').map(|s| s.trim())
          .filter(|s| !s.is_empty())   
          .map(|s| s.parse().unwrap()) 
          .collect();

    let of_vec: Vec<u64> = dep_info.offsets
          .split(';').map(|s| s.trim())
          .filter(|s| !s.is_empty())   
          .map(|s| s.parse().unwrap()) 
          .collect();
          
    let channel_ids = dep_info.channel_ids;
    
    let mut ret_vec:Vec<CompressInputImage> = vec![];

    let data_zero_idx = pkg_head_len + 8;
    
    for x in hit_vec
    {
        let (s_idx, e_idx) =  match x//当是第一个offset时候要特殊处理
                            {
                                0 => (data_zero_idx  as usize,                  (of_vec[x] + data_zero_idx) as usize),
                                k => ((of_vec[k - 1] + data_zero_idx) as usize, (of_vec[k] + data_zero_idx) as usize),
                            };

        let dpc_data:Vec<u8>  =  (&buf_v[s_idx .. e_idx]).to_vec();
                            
        let  ret =  CompressInputImage 
                    {
                        image_id: CString::new(src_ids_vec[x]).unwrap().into_raw(),
                        channel_id: CString::new(channel_ids.clone()).unwrap().into_raw(),
                        ts_ms: ts_vec[x],
                        buf: dpc_data.as_ptr(),
                        buf_len: (e_idx - s_idx) as u64,
                    };

        //write_a_file("test_rust.jpg".to_string(), &dpc_data);//测试:存成二进制文件
        std::mem::forget(dpc_data);
        ret_vec.push(ret);
    }

    *ret_num = ret_vec.len();
    let(ptr, _len, _cap) = ret_vec.into_raw_parts();
    ptr
}
