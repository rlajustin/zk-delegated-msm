use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    // This is where you have your .a and .h files
    let blst_lib_dir = env::var("BLST_LIB_DIR").unwrap_or_else(|_| "/tmp/blst".to_string());

    // 1. Tell the Rust Linker where to find libblst.a
    println!("cargo:rustc-link-search=native={}", blst_lib_dir);
    println!("cargo:rustc-link-lib=static=blst");
    println!("cargo:rustc-link-lib=m");

    let mut cfg = cc::Build::new();
    cfg.cpp(true);
    cfg.file("c/helper.cpp");
    cfg.include("c");
    cfg.include(&blst_lib_dir);
    cfg.include(format!("{}/bindings", blst_lib_dir));

    if target_os == "macos" {
        cfg.flag("-D__APPLE__");
    } else if target_os == "linux" {
        cfg.flag("-D__ELF__");
    }

    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-search=native=/opt/homebrew/opt/gmp/lib");
    println!("cargo:rustc-link-lib=gmp");
    cfg.include("/usr/local/include");
    cfg.include("/opt/homebrew/opt/gmp/include");
    // suppress C warnings
    cfg.flag("-w");

    cfg.compile("blst_custom");

    println!("cargo:rerun-if-changed=c/");
}
