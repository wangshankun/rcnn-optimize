
use std::collections::HashMap;

#[derive(Debug)]
struct Fud {
    keys:Vec<String>
}
fn foo(fud:&Fud){
    let mut results:HashMap<&String, Vec<f64>> = HashMap::new();
    for k in fud.keys.iter() {
        results.insert(k, Vec::new());
    }
    for v in fud.keys.iter() {
        results.get_mut(v).unwrap().push(0.0);
    }    
    println!(" {:?}", results);

    let key = "a".to_string(); 
    results.entry(&key).or_insert(Vec::new()).push(0.2);

    let key = "d".to_string(); 
    results.entry(&key).or_insert(Vec::new()).push(0.2);

    println!(" {:?}", results);
}

fn main() {
    let fud = Fud{keys:vec!["a".to_string(), "b".to_string(), "c".to_string()]};
    foo(&fud);
    println!(" {:?}", fud);
}
