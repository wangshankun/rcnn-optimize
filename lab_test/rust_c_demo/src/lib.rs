#![feature(core_intrinsics)]
#![feature(vec_into_raw_parts)]

#[warn(unused_variables)]
use std::ffi::{CStr, CString};
use std::slice;
use std::collections::HashMap;

#[allow(dead_code)]
#[allow(unused_unsafe)]
fn print_type_of<T>(_: &T)
{
    println!("{}", unsafe { std::intrinsics::type_name::<T>() });
}

#[repr(C)]
#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CompressInputImage {
    pub image_id:      *mut u8,
    pub channel_id:    *mut u8,
    pub ts_ms:         u64,
    pub buf:           *mut u8,
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

#[no_mangle]
pub unsafe extern "C" fn compress_images(cimgs:*mut CompressInputImage, len:usize, compress_rate:i32, image_format:i32, ret_num:*mut usize) 
-> *const CompressOutputData 
{

    let cm_img_array: &[CompressInputImage] = slice::from_raw_parts(cimgs, len as usize);
    //print_type_of(&cm_img_array);

    let mut channel_hit_list:HashMap<String, Vec<&CompressInputImage>> = HashMap::new();

    for x in cm_img_array
    {   //如果遇到不存在channel id那么就创建一个新的img收集器，把这个img放到收集器中；
        //如果存在就直接放到已经存channel id对应的收集器中
        let c_id = CStr::from_ptr(x.channel_id as *const i8).to_str().unwrap().to_owned();
        channel_hit_list.entry(c_id).or_insert(Vec::<&CompressInputImage>::new()).push(x);
    }

    //println!("{:?}",channel_hit_list);

    let mut ret_vec:Vec<CompressOutputData> = vec![];
    //打包，一个channel id hit打一包
    for (key, mut val) in channel_hit_list
    {
//         println!("channel id :{:?} have {} members", CString::from_raw(key as *mut i8), val.len());
         //让包内成员按照时间戳顺序从小到大排列
         val.sort_by(|a, b| a.ts_ms.cmp(&b.ts_ms));

         let ch_ids                 = key;
         let mut im_vec:Vec<String> = vec![];
         let mut ts_vec:Vec<u64>    = vec![];
         let mut of_vec:Vec<u64>    = vec![];
         let mut data:Vec<u8>       = vec![];
         let mut offset             = 0;

         for x in val
         {
              let a = CStr::from_ptr(x.image_id as *const i8).to_str().unwrap().to_owned();//*u8转字符串
              im_vec.push(a);
              ts_vec.push(x.ts_ms);
              //记录是下一个offset,即：这次的结束位置，最后一个就是整体长度
              //如果是视频压缩，那么offset就整体压缩视频长度
              offset = offset + x.buf_len;
              of_vec.push(offset);

              let img: Vec<u8> = Vec::from_raw_parts(x.buf, x.buf_len as usize, x.buf_len as usize);
              data.extend(img);
         }
         let im_ids = im_vec.join(";");
         let ts_ids_v:Vec<_> = ts_vec.iter().map(ToString::to_string).collect();
         let ts_ids = ts_ids_v.join(";");
         let of_ids_v:Vec<_> = of_vec.iter().map(ToString::to_string).collect();
         let of_ids = of_ids_v.join(";");

        //println!("打包  {:?} {:?} {:?} {} ",im_ids,ts_ids,of_ids,data.len());
        let ret = CompressOutputData
                  {
                      //channel_ids:*key as *const i8,//这种方式在C语言调用free(channel_ids)会崩掉
                      channel_ids:CString::new(ch_ids).unwrap().into_raw(),//key所有权转移到channel_ids，不用forget
                      image_ids:CString::new(im_ids).unwrap().into_raw(),//im_ids所有权已经转移到image_ids，不用forget
                      //ts_arrays:ts_ids.as_ptr() as *const i8,
                      ts_arrays:CString::new(ts_ids).unwrap().into_raw(),
                      //offsets:of_ids.as_ptr() as *const i8,
                      offsets:CString::new(of_ids).unwrap().into_raw(),
                      version:0,
                      compress_rate:compress_rate,
                      image_format:image_format,
                      buf:data.as_ptr(),
                      buf_len:offset,
                  };
         
//        std::mem::forget(ts_ids);
//        std::mem::forget(of_ids);
        std::mem::forget(data);//需要让rust忘记这段内存，不然指针返回给C语，而rust函数结束就释放内存了
        ret_vec.push(ret);
    }

   *ret_num = ret_vec.len();
   let(ptr, _len, _cap) = ret_vec.into_raw_parts();//采用ret_vec.as_ptr();会导致指针位置不对，不是最原始的数据指针
   ptr
}
