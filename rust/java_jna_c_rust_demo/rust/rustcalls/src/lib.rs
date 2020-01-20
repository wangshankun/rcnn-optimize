#![feature(core_intrinsics)]
#![feature(vec_into_raw_parts)]

use std::ffi::{CStr, CString};
use std::slice;
use std::ptr;
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
fn hit_index_vec(a:&Vec<&str>, b:&Vec<&str>) -> Vec<i32>
{
    let mut hit_idx:Vec<i32> = vec![];
    for x in b
    {
        let _ret = match (a).iter().position(|&r| &r == x)
        {
            Some(i) => hit_idx.push(i as i32), //C的解压库使用int类型
            None => {},//如果请求的id不在，那么忽略
        };
    }
    hit_idx
}


#[link(name="compress", kind="dylib")]
extern 
{
    fn compress_2_h265(in_datas:*mut u8, sizes:*mut i32, in_num:i32, out_data:*mut *mut u8, out_size:*mut i32) -> i32;
}
#[link(name="decompress", kind="dylib")]
extern
{
    fn decompress_2_jpeg(video_buf:*const u8, sizes:i32, hit_array:*const i32, hit_len:i32, out_buf_arry:*mut *mut u8, out_size_arry:*mut *mut i32);
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
         let mut sizes:Vec<i32>     = vec![];

         for x in val
         {
              let a = CStr::from_ptr(x.image_id).to_str().unwrap().to_owned();//*u8转字符串
              im_vec.push(a);
              ts_vec.push(x.ts_ms);
              offset = offset + x.buf_len;
              of_vec.push(offset);

              let img: Vec<u8> = Vec::from_raw_parts(x.buf as *mut _, x.buf_len as usize, x.buf_len as usize);
              data.extend(img);
              sizes.push(x.buf_len as i32);
         }
         let im_ids = im_vec.join(";");
         let ts_ids_v:Vec<_> = ts_vec.iter().map(ToString::to_string).collect();
         let ts_ids = ts_ids_v.join(";");
         let of_ids_v:Vec<_> = of_vec.iter().map(ToString::to_string).collect();
         let of_ids = of_ids_v.join(";");

        //将图片压缩后，返回压缩内存
        let mut out_data_ptr:*mut u8 = ptr::null_mut();
        let mut out_size:i32         = 0;
        compress_2_h265(data.as_mut_ptr(), sizes.as_mut_ptr(), sizes.len() as i32,
                        &mut out_data_ptr, &mut out_size);
        let mut out_data:Vec<u8>  = Vec::from_raw_parts(out_data_ptr as *mut _ , out_size as usize, out_size as usize);
        let cmpress_data_len = out_data.len() as u64;
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
                      buf_len:cmpress_data_len,//有效数据的长度
                  };  
        let ser_cod = serde_json::to_string(&cod).unwrap();
        //println!("{}", ser_cod);
        let mut ser_vec_char = ser_cod.into_bytes();
        //to_le_bytes() x86和网络字节序默认小端机器;因此就设置长度为小端存储
        //这里把usize强转为i64也是为了适应32/64位机器，usize长度不固定/java没有unsigned long选用long
        let mut ser_head_len:Vec<u8> = (ser_vec_char.len() as i64).to_le_bytes().to_vec();
        ser_data.append(&mut ser_head_len);//最开始8个字节为序列化的长度
        ser_data.append(&mut ser_vec_char);//接下来是序列化包头
        ser_data.append(&mut out_data);//最后是压缩的数据
        
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
                      buf_len:cmpress_data_len,//有效数据的长度
                  };
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
    let data_zero_idx = (pkg_head_len + 8) as usize;
    let pkg_head = String::from_utf8_lossy(&buf_v[8..data_zero_idx]);//将序列化二进制内存转成字符串

    let dep_info: SerializeCOData = serde_json::from_str(&pkg_head).unwrap();//反序列化

    let hit_img_ids = CStr::from_ptr(hit_ids).to_str().unwrap().to_owned();
    let req_ids_vec: Vec<&str> = hit_img_ids.split(";").collect();
    let src_ids_vec: Vec<&str> = dep_info.image_ids.split(";").collect();
    let hit_vec = hit_index_vec(&src_ids_vec, &req_ids_vec);

    let ts_vec: Vec<u64> = dep_info.ts_arrays
          .split(';').map(|s| s.trim())
          .filter(|s| !s.is_empty())   
          .map(|s| s.parse().unwrap()) 
          .collect();
          
    let channel_ids = dep_info.channel_ids;
    let mut ret_vec:Vec<CompressInputImage> = vec![];

    let video_data:Vec<u8>          = (&buf_v[data_zero_idx..]).to_vec();
    let mut out_buf_arry:  *mut u8  = ptr::null_mut();
    let mut out_size_arry: *mut i32 = ptr::null_mut();
    //解压成jpeg图片数组
    decompress_2_jpeg(video_data.as_ptr(), video_data.len() as i32, hit_vec.as_ptr(), hit_vec.len() as i32, &mut out_buf_arry, &mut out_size_arry);
    std::mem::forget(video_data);//这部分内存由decompress_2_jpeg函数中的AVIO流接口的析构函数释放

    let out_size_vec:Vec<i32>  = Vec::from_raw_parts(out_size_arry as *mut _ ,  hit_vec.len(),  hit_vec.len());
    let out_sum_len:i32 = (&out_size_vec).into_iter().sum(); 
    let out_buf_vec:Vec<u8>  = Vec::from_raw_parts(out_buf_arry as *mut _ , out_sum_len as usize, out_sum_len as usize);

    let mut cur_size:usize = 0;
    for (indx, m_size) in out_size_vec.iter().enumerate()
    {
        let x = hit_vec[indx] as usize;
        
        let dpc_data:Vec<u8>  =  (&out_buf_vec[cur_size .. (cur_size + *m_size as usize)]).to_vec();
        cur_size =  cur_size + *m_size as usize;

        let  ret =  CompressInputImage
                    {
                        image_id: CString::new(src_ids_vec[x]).unwrap().into_raw(),
                        channel_id: CString::new(channel_ids.clone()).unwrap().into_raw(),
                        ts_ms: ts_vec[x],
                        buf: dpc_data.as_ptr(),
                        buf_len: *m_size as u64,
                    };

        //write_a_file("test_rust.jpg".to_string(), &dpc_data);//测试:存成二进制文件
        std::mem::forget(dpc_data);
        ret_vec.push(ret);
    }

    *ret_num = ret_vec.len();
    let(ptr, _len, _cap) = ret_vec.into_raw_parts();
    ptr
}
