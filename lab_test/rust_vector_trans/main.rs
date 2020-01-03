#![feature(core_intrinsics)]

fn print_type_of<T>(_: &T) {
    println!("{}", unsafe { std::intrinsics::type_name::<T>() });
}


fn print_matrix(vec: &Vec<Vec<f64>>) {
    for row in vec {
        let cols_str: Vec<_> = row.iter().map(ToString::to_string).collect();
        let line = cols_str.join("\t");
        println!("{}", line);
    }
}

fn main() {
//vector数字转字符串
//https://users.rust-lang.org/t/how-do-i-convert-vec-of-i32-to-string/18669/8

    let stuff = vec![5, 5, 2, 0, 8];
    /*
    let stuff_str: String = stuff
        .into_iter()
        .map(|d| std::char::from_digit(d, 10).unwrap())
        .collect();
    println!("{}", stuff_str);
    */

/*
   let stuff_str: String = stuff.into_iter().map(|i| i.to_string()+";").collect::<String>();
   println!("{}", stuff_str);
*/
   let stuff_str:Vec<_> = stuff.iter().map(ToString::to_string).collect();
   let ss = stuff_str.join(";");
   println!("{}",ss);
   print_type_of(&ss);

//字符串 转数字 vector
//https://stackoverflow.com/questions/34090639/how-do-i-convert-a-vector-of-strings-to-a-vector-of-integers-in-a-functional-way/34090825
    let input = "1,2,3";
    //let numbers: Result<Vec<u16>, _> = input.split(",").map(|x| x.parse()).collect();
    //println!("{:?}", numbers);
    let bar: Vec<u16> = input.split(",").map(|x| x.parse::<u16>().unwrap()).collect();
    println!("{:?}", bar);
   

    let matrix:Vec<Vec<f64>> = vec![vec![1.212,323.12,12.12],vec![0.001,2990.9,14.623],vec![0.98,77.6,1314.521]];
    print_matrix(&matrix);

}
