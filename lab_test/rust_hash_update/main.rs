
use std::collections::HashMap;

fn main() {
    let mut my_map = HashMap::new();
    my_map.insert("a", 1);
    my_map.insert("b", 3);
    *my_map.entry("a").or_insert(42) += 10;
    *my_map.entry("c").or_insert(42) += 10;

    println! {"{:?}", my_map};

    let mut vec_map = HashMap::new();
    
    vec_map.insert("a", Vec::<u8>::new());

    println! {"{:?}", vec_map};
}

