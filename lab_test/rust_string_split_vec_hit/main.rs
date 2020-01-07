fn hit_index_vec(a:&Vec<&str>, b:&Vec<&str>) -> Vec<usize> 
{
    let mut hit_idx:Vec<usize> = vec![];
    for x in b
    {
        let _ret = match (a).iter().position(|&r| &r == x)
        {
            Some(i) => hit_idx.push(i),
            None => {},
        };
    }
    hit_idx
}

fn main()
{
    let a = "100;101;102;103;104;105".split(";");
    let b = "101;104".split(";");
    let c = "101;108".split(";");

    let a_vec: Vec<&str> = a.collect();
    let b_vec: Vec<&str> = b.collect();
    let c_vec: Vec<&str> = c.collect();

    println!("{:?} {:?}",a_vec, b_vec);

    println!("{}",(&a_vec).into_iter().any(|v| v == &b_vec[0]));
    println!("{}",(&a_vec).iter().position(|&r| &r == &b_vec[1]).unwrap());
    println!("{:?}",(&a_vec).iter().position(|&r| &r == &b_vec[1]));

    println!("{:?}",hit_index_vec(&a_vec, &b_vec));
   
    let indx = hit_index_vec(&a_vec, &c_vec);
    println!("{:?}",hit_index_vec(&a_vec, &c_vec));
    println!("is all elem contains? {}", (c_vec.len()==indx.len()));
}
