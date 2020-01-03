use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Person {
    name: String,
    age: u8,
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
}

