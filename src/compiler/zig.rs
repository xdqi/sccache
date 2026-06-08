// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Support for `zig cc` / `zig c++` as a first-class distributable compiler.
//!
//! Design (see docs/2026-06-08-zig-cc-dist-design.md in distcc-action):
//!   - `zig cc` / `zig c++` IS a clang driver, so all command-line semantics
//!     (argument parsing, preprocessing, the dist compile command) are delegated
//!     to an embedded `Clang` impl.
//!   - zig gets its OWN identity (`CCompilerKind::Zig`) and its OWN toolchain
//!     packaging (bundle the `zig` binary + the entire `lib/` tree, like rustc's
//!     sysroot — NOT gcc-style ldd + per-tool discovery).
//!   - The remote executable is the real `zig` binary with `cc`/`c++` as the
//!     first argument (zig dispatches on argv[1]; there is no argv0 multicall).
//!   - Two non-transparent tweaks vs clang, both verified empirically:
//!       * prepend the `cc`/`c++` subcommand to the argument list;
//!       * strip the `-x <*-cpp-output>` language pair, since `zig cc` rejects
//!         those language names (only `-x c` or the `.i` extension work); the
//!         `.i` extension drives the language instead.
//!   - Like rustc, inject an explicit host `-target` when the user gave none, so
//!     the remote (possibly different-default) zig produces host-compatible
//!     objects.

#![allow(unused_imports, dead_code, unused_variables)]

use crate::compiler::c::{CCompilerImpl, CCompilerKind, ParsedArguments};
use crate::compiler::clang::{self, Clang};
use crate::compiler::{
    CCompileCommand, Cacheable, CompileCommand, CompilerArguments, Language,
};
use crate::mock_command::CommandCreatorSync;
use crate::dist;
use async_trait::async_trait;
use log::Level::Trace;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process;

use crate::errors::*;

/// A `CCompilerImpl` for `zig cc` / `zig c++`. Delegates clang semantics to an
/// embedded `Clang`, but identifies as `Zig` and rewrites the emitted commands
/// to invoke `zig <subcmd>`.
#[derive(Clone, Debug)]
pub struct Zig {
    /// The embedded clang impl all command-line semantics delegate to.
    pub clang: Clang,
}

impl Zig {
    /// The zig subcommand to invoke: `c++` for the C++ driver, else `cc`.
    fn subcommand(&self) -> &'static str {
        if self.clang.plusplus() { "c++" } else { "cc" }
    }
}

/// True if `arg` is a `-x` language name that `zig cc` rejects (the
/// `*-cpp-output` family clang/gcc use to mark locally-preprocessed input).
fn is_cpp_output_lang(arg: &str) -> bool {
    arg.ends_with("cpp-output")
}

/// Rewrite the dist command's flat `String` argument vector for zig:
///   1. drop any `-x <*-cpp-output>` pair (zig rejects those names; the `.i`
///      extension carries the language),
///   2. silence `-Wgnu-line-marker` on the locally-preprocessed input,
///   3. prepend the `cc`/`c++` subcommand.
///
/// We do NOT inject a `-target`: zig defaults to the native target (with native
/// CPU features), and the remote builder is the same architecture. Injecting an
/// explicit triple like `x86_64-linux-gnu` would reset the CPU baseline and
/// break code that uses native features (e.g. redis/xxhash AVX intrinsics). A
/// user-supplied `-target` is passed through untouched (zig cc's cross use).
fn rewrite_dist_arguments(subcommand: &str, arguments: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(arguments.len() + 2);
    let mut it = arguments.into_iter().peekable();
    while let Some(arg) = it.next() {
        if arg == "-x" {
            // Peek the language operand; drop the pair if it's *-cpp-output.
            if let Some(next) = it.peek() {
                if is_cpp_output_lang(next) {
                    let _ = it.next(); // consume the language name
                    continue;
                }
            }
            out.push(arg);
            continue;
        }
        out.push(arg);
    }
    // sccache hands the remote compiler a locally-preprocessed file full of
    // `# <line> "file"` GNU line markers. zig's bundled clang warns on those
    // (-Wgnu-line-marker); with the user's -Werror that turns a sccache-internal
    // artifact into a hard error (e.g. redis/hiredis builds -Werror). The marker
    // is not in the user's source, so silence just that warning on the remote
    // compile. (Native clang sidesteps this via -frewrite-includes when
    // rewrite_includes_only is on; we keep it robust regardless of that mode.)
    out.push("-Wno-gnu-line-marker".to_string());
    // The subcommand must be argv[0] (zig dispatches on it).
    out.insert(0, subcommand.to_string());
    out
}

/// Local-command counterpart of `rewrite_dist_arguments`. The local command runs
/// the user's real preprocessing/compile, so it only needs the subcommand
/// prepended -- no `-x` stripping (the local path uses the real input) and no
/// line-marker suppression.
fn rewrite_local_arguments(subcommand: &str, arguments: Vec<OsString>) -> Vec<OsString> {
    let mut out: Vec<OsString> = Vec::with_capacity(arguments.len() + 1);
    out.push(OsString::from(subcommand));
    out.extend(arguments);
    out
}

#[async_trait]
impl CCompilerImpl for Zig {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Zig
    }
    fn plusplus(&self) -> bool {
        self.clang.plusplus()
    }
    fn version(&self) -> Option<String> {
        self.clang.version()
    }

    fn parse_arguments(
        &self,
        arguments: &[OsString],
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
    ) -> CompilerArguments<ParsedArguments> {
        // `arguments` still begins with the zig subcommand (`cc`/`c++`) because
        // sccache split the command as exe=`zig`, args=`[cc, ...]`. Strip it so
        // the clang parser sees a plain clang command line; we re-add the
        // subcommand when generating the compile commands.
        let stripped: &[OsString] = match arguments.first().and_then(|a| a.to_str()) {
            Some("cc") | Some("c++") => &arguments[1..],
            _ => arguments,
        };
        self.clang.parse_arguments(stripped, cwd, env_vars)
    }

    #[allow(clippy::too_many_arguments)]
    async fn preprocess<T>(
        &self,
        creator: &T,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        may_dist: bool,
        rewrite_includes_only: bool,
        preprocessor_cache_mode: bool,
    ) -> Result<process::Output>
    where
        T: CommandCreatorSync,
    {
        use crate::mock_command::{CommandCreator, RunCommand};
        use crate::util::run_input_output;

        // Local preprocessing must run `zig <subcmd> -E ...`. clang/gcc's
        // preprocess builds the command from `executable` + appended args, with
        // no way to inject argv[0]. So we build the command ourselves: start on
        // the zig binary, push the cc/c++ subcommand, then reuse gcc's
        // `preprocess_cmd` to append exactly the args clang would (using clang's
        // language mapping and whitespace flags), and run it.
        let mut ignorable_whitespace_flags = if preprocessor_cache_mode {
            vec![]
        } else {
            vec!["-P".to_string()]
        };
        // Mirror Clang::preprocess: clang>=14 supports -fminimize-whitespace,
        // except on assembler-with-cpp. zig bundles a recent clang, so enable it
        // (skip for assembly, matching clang.rs).
        if parsed_args.language != Language::AssemblerToPreprocess {
            ignorable_whitespace_flags.push("-fminimize-whitespace".to_string());
        }

        let mut cmd = creator.clone().new_command_sync(executable);
        cmd.arg(self.subcommand());
        crate::compiler::gcc::preprocess_cmd(
            &mut cmd,
            parsed_args,
            cwd,
            env_vars,
            may_dist,
            // Use Clang's behavior for the preprocessor command.
            CCompilerKind::Clang,
            rewrite_includes_only,
            ignorable_whitespace_flags,
            clang::language_to_clang_arg,
        );
        if log_enabled!(Trace) {
            trace!("zig preprocess: {:?}", cmd);
        }
        run_input_output(cmd, None).await
    }

    fn generate_compile_commands<T>(
        &self,
        path_transformer: &mut dist::PathTransformer,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        rewrite_includes_only: bool,
    ) -> Result<(
        Box<dyn CompileCommand<T>>,
        Option<dist::CompileCommand>,
        Cacheable,
    )>
    where
        T: CommandCreatorSync,
    {
        // Delegate to gcc's generator (the shared C/C++ command builder) using
        // clang's language mapping -- exactly what Clang::generate_compile_commands
        // does -- but get back the concrete SingleCompileCommand so we can rewrite
        // both the local and dist commands for zig.
        let (mut local_cmd, dist_cmd, cacheable) =
            crate::compiler::gcc::generate_compile_commands(
                path_transformer,
                executable,
                parsed_args,
                cwd,
                env_vars,
                // Use Clang as the kind for command generation so we get clang's
                // behavior (no -fpreprocessed, which is a gcc-only branch).
                CCompilerKind::Clang,
                rewrite_includes_only,
                clang::language_to_clang_arg,
            )?;

        let subcmd = self.subcommand();

        // Rewrite the LOCAL command: zig <subcmd> <args...>.
        local_cmd.arguments = rewrite_local_arguments(subcmd, local_cmd.arguments);

        // Rewrite the DIST command (if dist is allowed for this compile).
        let dist_cmd = dist_cmd.map(|mut dc| {
            dc.arguments = rewrite_dist_arguments(subcmd, dc.arguments);
            dc
        });

        Ok((CCompileCommand::new(local_cmd), dist_cmd, cacheable))
    }
}

/// Parse a single `.field = "value",` line out of `zig env` ZON output.
/// `zig env` prints e.g. `    .lib_dir = "/opt/zig/lib",` per line.
fn zig_env_field(zig_env_output: &str, field: &str) -> Option<String> {
    let needle = format!(".{} = \"", field);
    for line in zig_env_output.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&needle) {
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
    }
    None
}

/// Toolchain packager for zig: bundle the `zig` binary plus its entire `lib/`
/// tree (libc headers, compiler-rt, std, per-target sources). zig compiles
/// on-demand from these sources, so the whole tree must travel together --
/// exactly the rustc-sysroot model (`add_dir_contents`), NOT gcc-style ldd +
/// per-tool discovery.
#[cfg(feature = "dist-client")]
pub struct ZigToolchainPackager {
    /// Path to the `zig` executable.
    pub executable: std::path::PathBuf,
}

#[cfg(feature = "dist-client")]
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
impl crate::dist::pkg::ToolchainPackager for ZigToolchainPackager {
    fn write_pkg(self: Box<Self>, f: fs_err::File) -> Result<()> {
        use crate::dist::pkg;
        info!("Packaging zig toolchain {}", self.executable.display());

        // Discover the lib tree via `zig env`.
        let output = process::Command::new(&self.executable)
            .arg("env")
            .output()
            .context("Failed to run `zig env` to locate the zig lib dir")?;
        if !output.status.success() {
            bail!("`zig env` failed with status {}", output.status);
        }
        let env_str = String::from_utf8(output.stdout).context("`zig env` output not UTF-8")?;
        let lib_dir = zig_env_field(&env_str, "lib_dir")
            .context("Could not find lib_dir in `zig env` output")?;

        let mut package_builder = pkg::ToolchainPackageBuilder::new();
        package_builder.add_common()?;
        // The zig binary itself (+ any ldd deps; zig is typically static so this
        // is usually just the binary).
        package_builder.add_executable_and_deps(self.executable.clone())?;
        // The entire lib/ tree -- the rustc-style wholesale bundle.
        package_builder.add_dir_contents(std::path::Path::new(&lib_dir))?;

        package_builder.into_compressed_tar(f)
    }
}
