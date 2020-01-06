#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
struct MyStruct {
    foo: u64,
    tsk: u16,
    bar: u8,
}

//int n = 1;
// little endian if true
//if(*(char *)&n == 1) {...}
fn is_little_end() -> bool
{
    let i:i32 = 1;
    let te:i8 = *&i as i8;
    //println!("{} {}", file!(), line!()); 
    te == 1
}

fn main() {
    let v: Vec<u8> = vec![52,0,0,0,0,0,0,0,1,2,3];
    let s: MyStruct = unsafe { std::ptr::read(v.as_ptr() as *const _) };
    println!("here is the struct: {:?}", s);
    println!("is little end: {}",is_little_end());
    let k:u64 = 52;
    println!("little end {:?}",k.to_le_bytes().to_vec());
    println!("big end {:?}",k.to_be_bytes().to_vec());
    println!("net end {:?}",k.to_ne_bytes().to_vec());
}
