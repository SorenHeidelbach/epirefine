//! Records the commit the extension is built from as EPIREFINE_RS_COMMIT (module attribute `__commit__`), so that
//! a run record names the build even when the version did not change. Order: the environment variable
//! EPIREFINE_RS_COMMIT (a CI job or a source build that knows its commit), else `git rev-parse HEAD` of the
//! EpiRefine repository (only when this crate sits in its `rust/` directory) with "-dirty" appended when tracked
//! files differ from it, else nothing (an sdist without .git:
//! `__commit__` is then None).
use std::path::PathBuf;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    println!("cargo:rerun-if-env-changed=EPIREFINE_RS_COMMIT");
    println!("cargo:rerun-if-changed=build.rs");
    // the package files (a change of them changes "-dirty"), and git's HEAD, index and the branch ref (a commit)
    for p in ["src", "Cargo.toml", "Cargo.lock", "../pyproject.toml"] {
        println!("cargo:rerun-if-changed={p}");
    }
    let commit = match std::env::var("EPIREFINE_RS_COMMIT") {
        Ok(c) if !c.trim().is_empty() => Some(c.trim().to_string()),
        _ => {
            // only this repository's own commit: an sdist unpacked inside another git tree (a conda build) has none
            let manifest = std::env::var("CARGO_MANIFEST_DIR").ok().map(PathBuf::from);
            let own = git(&["rev-parse", "--show-toplevel"]).and_then(|t| std::fs::canonicalize(t).ok())
                == manifest.as_ref().and_then(|d| d.parent()).and_then(|d| std::fs::canonicalize(d).ok())
                && manifest.as_ref().and_then(|d| d.file_name()).map(|n| n == "rust").unwrap_or(false);
            let head = if own { git(&["rev-parse", "HEAD"]) } else { None };
            if head.is_some() {
                for what in ["HEAD", "index"] {
                    if let Some(p) = git(&["rev-parse", "--git-path", what]) {
                        println!("cargo:rerun-if-changed={}", PathBuf::from(p).display());
                    }
                }
                if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
                    if let Some(p) = git(&["rev-parse", "--git-path", &r]) {
                        println!("cargo:rerun-if-changed={}", PathBuf::from(p).display());
                    }
                }
            }
            head.map(|h| {
                let dirty = Command::new("git")
                    .args(["diff-index", "--quiet", "HEAD", "--"])
                    .status()
                    .map(|s| !s.success())
                    .unwrap_or(false);
                if dirty { format!("{h}-dirty") } else { h }
            })
        }
    };
    if let Some(c) = commit {
        println!("cargo:rustc-env=EPIREFINE_RS_COMMIT={c}");
    }
}
