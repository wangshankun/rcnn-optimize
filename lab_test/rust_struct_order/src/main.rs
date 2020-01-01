use std::cmp::Ordering;
use rand::Rng; //导入外部的包... 记得修改toml文件

//保证age是可比较的
pub struct Person<T : std::cmp::PartialOrd> {
    age: T,
}

//注意泛型T的位置
impl<T> Person<T> where T:std::cmp::PartialOrd{
    //也可impl<T:std::cmp::PartialOrd> Person<T>
    pub fn new(a:T) -> Self{
        Person {age:a}
    }

}
//让Person可比较大小, 操作符重载???
impl<T:std::cmp::PartialOrd> PartialOrd for Person<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.age.partial_cmp(&other.age)
    }
}
//让Person可比较是否相等, 操作符重载???
impl<T:std::cmp::PartialOrd> PartialEq for Person<T> {
    fn eq(&self, other: &Self) -> bool {
        self.age == other.age
    }
}

/*
--排序的Vec<T>中的T
--跟Person<T>的T 之间不一样
*/
pub fn quicksort<T>( arr : &mut Vec<T>)where T:std::cmp::PartialOrd {
    quick_sorted(arr,0,arr.len()-1);
}
fn quick_sorted<T>( arr:&mut Vec<T>, a:usize,b : usize)where T:std::cmp::PartialOrd {
    if a<b {
        if b-a < 20 {
            insert_sorted(arr,a,b);
        }else{
            let p = partion(arr,a,b);
            if p !=0 {
                quick_sorted(arr,a,p-1); //无符号整数...越界
            }
            quick_sorted(arr,p+1,b);
        }
    }
}
//注意A的写法... vec下标为 usize, 容易越界, 不过发现更多的小错误...
fn partion<T>( arr :&mut Vec<T>, p:usize,r:usize)->usize  where T:std::cmp::PartialOrd { 
    let mut i = p;
    for j in p..r {
        if arr[j] < arr[r] { //比较T需要加std::cmp::PartialOrd
            arr.swap(i, j);
            i+=1;
        }
    }
    arr.swap(i,r);
    i
}

//插入排序
fn insert_sorted<T>( arr :&mut Vec<T>,l:usize,r:usize) where T:std::cmp::PartialOrd {
    let mut i:usize;
    for j in l+1..r+1 {
        i = j;
        while i>l && arr[i] < arr[i-1] {
            arr.swap(i,i-1);
            i-=1;
        }
    }
}

fn main() {
    let mut rag = rand::thread_rng();
    let mut a  = vec![]; //暂时不指定类型
    for _j in 0..10 {
        let tmp:u8 = rag.gen(); //产生随机数
        let person = Person::new(tmp); //创建结构体
        a.push(person);
    }
    quicksort(&mut a);
    for i in &a { //引用, 不释放空间
        println!("{:#?}",i.age)
    }

}
//原文链接：https://blog.csdn.net/qq_43239441/article/details/103744538
