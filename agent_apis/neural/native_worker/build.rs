fn main() {
    println!(
        "cargo:rustc-link-search=native={}/native",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    println!("cargo:rustc-link-lib=static=npyrandom");
    println!("cargo:rustc-link-lib=m");
}
