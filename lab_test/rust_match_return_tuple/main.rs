
fn main() {
    let v: Vec<i32> = vec![52,0,0,0,0,0,0,0,1,2,3];
    let x:usize = 1;
    let (buf_len, data) =  match x
                         {
                                0 => (0, v[0]),
                                k => (k, v[k] - v[k - 1]),
                         };

    println!("{} {} ",buf_len, data);
}
