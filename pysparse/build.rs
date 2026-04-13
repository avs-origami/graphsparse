use std::env;
use std::process::Command;

fn main() {
    let cwd = env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed={cwd}/../tensorviz/tensorviz.cpp");
    println!("cargo:rerun-if-changed={cwd}/../tensorviz/setup.py");
    println!("cargo:rerun-if-changed={cwd}/../tensorviz/build.sh");
    println!("cargo:warning=Rebuilding {cwd}/../tensorviz/tensorviz.cpp");

    let out = Command::new(format!("{cwd}/../tensorviz/build.sh"))
        .output()
        .unwrap();

    let sout = String::from_utf8(out.stdout).unwrap();
    let serr = String::from_utf8(out.stderr).unwrap();

    // for line in sout.lines() {
    //     println!("cargo:warning={line}");
    // }

    for line in serr.lines() {
        println!("cargo:warning={line}");
    }

    assert!(out.status.success())
}