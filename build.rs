use std::env;
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let blst_lib_dir = env::var("BLST_LIB_DIR").unwrap_or_else(|_| "/tmp/blst".to_string());

    let mut cfg = cc::Build::new();
    cfg.cpp(true)
        .flag("-w") // Suppress warnings
        .file("c/toeplitz.cpp")
        .file("c/trapdoor.cpp")
        .include("c")
        .include(&blst_lib_dir)
        .include(format!("{}/bindings", blst_lib_dir));

    // --- OS Specific Macros ---
    if target_os == "macos" {
        cfg.flag("-D__APPLE__");
    } else {
        cfg.flag("-D__ELF__");
    }

    // --- Link blst ---
    println!("cargo:rustc-link-search=native={}", blst_lib_dir);
    println!("cargo:rustc-link-lib=static=blst");

    // --- Dependency Management (GMP, OpenSSL, NTL) ---
    if target_os == "macos" {
        // Use brew to find prefix paths
        let prefixes = ["gmp", "openssl@3", "ntl"];
        for lib in prefixes {
            if let Ok(output) = Command::new("brew").args(["--prefix", lib]).output() {
                let prefix = std::str::from_utf8(&output.stdout).unwrap().trim();
                println!("cargo:rustc-link-search=native={}/lib", prefix);
                cfg.include(format!("{}/include", prefix));
            }
        }
    } else {
        // On Linux, standard locations (/usr/include, /usr/local/include) are usually searched by default.
        // But we add them explicitly just in case.
        cfg.include("/usr/local/include");
        cfg.include("/usr/include");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=/usr/lib");
    }

    // --- Final Linker Instructions ---
    // Note: Link order matters for some compilers.
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=crypto"); // OpenSSL
    println!("cargo:rustc-link-lib=m"); // Math lib

    cfg.compile("blst_custom");

    println!("cargo:rerun-if-changed=c/");
}

// use std::env;
//
// fn main() {
//     let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
//
//     // This is where you have your .a and .h files
//     let blst_lib_dir = env::var("BLST_LIB_DIR").unwrap_or_else(|_| "/tmp/blst".to_string());
//
//     // 1. Tell the Rust Linker where to find libblst.a
//     println!("cargo:rustc-link-search=native={}", blst_lib_dir);
//     println!("cargo:rustc-link-lib=static=blst");
//     println!("cargo:rustc-link-lib=m");
//
//     let mut cfg = cc::Build::new();
//     cfg.cpp(true);
//     cfg.file("c/toeplitz.cpp");
//     cfg.file("c/trapdoor.cpp");
//     cfg.include("c");
//     cfg.include(&blst_lib_dir);
//     cfg.include(format!("{}/bindings", blst_lib_dir));
//
//     if target_os == "macos" {
//         cfg.flag("-D__APPLE__");
//     } else if target_os == "linux" {
//         cfg.flag("-D__ELF__");
//     }
//
//     println!("cargo:rustc-link-lib=ntl");
//     println!("cargo:rustc-link-search=native=/opt/homebrew/opt/gmp/lib");
//     println!("cargo:rustc-link-lib=gmp");
//
//     #[cfg(target_os = "macos")]
//     {
//         let output = std::process::Command::new("brew")
//             .args(["--prefix", "openssl"])
//             .output()
//             .expect("Failed to run brew");
//         let prefix = std::str::from_utf8(&output.stdout).unwrap().trim();
//         println!("cargo:rustc-link-search=native={}/lib", prefix);
//         println!("cargo:include={}/include", prefix);
//     }
//     println!("cargo:rustc-link-lib=crypto");
//     cfg.include("/usr/local/include");
//     cfg.include("/opt/homebrew/opt/gmp/include");
//     cfg.include("/opt/homebrew/Cellar/openssl@3/3.6.1/include");
//
//     // suppress C warnings
//     cfg.flag("-w");
//
//     cfg.compile("blst_custom");
//
//     println!("cargo:rerun-if-changed=c/");
// }
