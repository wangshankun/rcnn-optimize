fn main() {
    println!("cargo:rustc-link-search=native=../jpg_2_h265/");
    println!("cargo:rustc-link-lib=dylib=compress");
    println!("cargo:rustc-link-search=native=../h265_2_jpg/");
    println!("cargo:rustc-link-lib=dylib=decompress");
}
