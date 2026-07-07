"""Best-practice rules for Rust edits, selected by relevance at inject time.

See :mod:`._rules` for the model. ``core`` rules are always emitted; the rest
gate on trigger substrings found in the task context.
"""

from ._rules import Rule

RUST_RULES = [
    # --- core baseline (always emitted) ---
    Rule(
        "Own at boundaries, borrow inside: take &str/&[T]/&Path, return owned; Cow<'_,T> for sometimes-owned; Rc<RefCell<T>> only for real shared graphs.",
        core=True,
    ),
    Rule(
        "No unwrap/expect/panic outside tests & provably-infallible paths → if the invariant is real, encode it in the type; propagate with ? + Result.",
        core=True,
    ),
    Rule(
        "Make illegal states unrepresentable: newtype distinct values (struct UserId(u64), not a type alias); enum > a struct of several Options.",
        core=True,
    ),
    Rule(
        "Must pass: clippy -D warnings, rustfmt. DRY via traits/generics/blanket impls; delete dead code & unused deps (cargo machete).",
        core=True,
    ),
    # --- error handling ---
    Rule(
        "thiserror(lib)/anyhow(app); #[from]/#[source] keep the error chain; never Result<_,String> or Box<dyn Error> in a public API.",
        triggers=(
            "error",
            "result",
            "thiserror",
            "anyhow",
            "unwrap",
            "expect",
            "panic",
            "fail",
            "?",
        ),
    ),
    # --- API / trait design ---
    Rule(
        "Small composed traits; generics + impl Trait > dyn. #[non_exhaustive]/#[must_use]; tightest visibility (pub(crate)/pub(super)); semver: field/variant/bound additions break (cargo semver-checks).",
        triggers=(
            "pub ",
            "trait",
            "impl trait",
            "dyn ",
            "api",
            "sealed",
            "non_exhaustive",
            "must_use",
            "semver",
            "generic",
            "<t",
        ),
    ),
    # --- performance / allocation / layout ---
    Rule(
        "Measure first (criterion/samply/dhat). Avoid needless .clone(); reuse buffers (Vec::clear), with_capacity, SmallVec, Box<[T]>, bumpalo arena; don't .collect() mid-pipeline. Release: lto + codegen-units=1; zero-copy parse (winnow/zerocopy/bytes). Layout: struct-of-arrays, box large enum variants, #[repr(align(64))] vs false sharing.",
        triggers=(
            "perf",
            "hot",
            "alloc",
            "clone",
            "loop",
            "iterator",
            "collect",
            "capacity",
            "vec",
            "bench",
            "simd",
            "buffer",
            "cache",
            "throughput",
            "latency",
        ),
    ),
    # --- concurrency / async ---
    Rule(
        "Message-passing > shared state; Arc<Mutex> only for real shared state; bounded channels (unbounded = leak). Never hold a lock across .await; no detached tokio::spawn → JoinSet; select! arms must be cancel-safe. rayon for CPU-bound, async for I/O, spawn_blocking for CPU in async.",
        triggers=(
            "async",
            "await",
            "tokio",
            "thread",
            "spawn",
            "mutex",
            "arc<",
            "channel",
            "lock",
            "rayon",
            "atomic",
            "sync",
            "send",
            "concurren",
            "parallel",
        ),
    ),
    # --- unsafe / FFI ---
    Rule(
        "Every unsafe block → a // SAFETY: comment stating its invariants; wrap unsafe in safe modules; cargo miri for unsafe/FFI; bytemuck/zerocopy > hand-rolled transmute; #[repr(C)] + coarse-grained batching across FFI.",
        triggers=(
            "unsafe",
            "transmute",
            "ptr",
            "ffi",
            "extern",
            "repr(c",
            "maybeuninit",
            "raw",
            "libc",
            "bindgen",
        ),
    ),
    # --- security / crypto ---
    Rule(
        "subtle ConstantTimeEq for secret comparison (== leaks timing); zeroize/ZeroizeOnDrop for keys/nonces; secrecy::Secret + a redacting Debug; use RustCrypto/ring/dalek, don't hand-roll primitives.",
        triggers=(
            "crypto",
            "key",
            "secret",
            "nonce",
            "hash",
            "sign",
            "verify",
            "password",
            "token",
            "zeroize",
            "constant-time",
            "subtle",
            "cipher",
            "hmac",
            "encrypt",
        ),
    ),
    # --- testing ---
    Rule(
        "proptest/quickcheck for laws & round-trips (parse(serialize(x))==x); insta snapshots; criterion benches; cargo-fuzz every parser/verifier; test vectors + fixed RNG seeds.",
        triggers=(
            "test",
            "proptest",
            "quickcheck",
            "fuzz",
            "criterion",
            "assert",
            "bench",
            "mock",
        ),
    ),
    # --- WebAssembly target (Rust → WASM) ---
    Rule(
        "WASM: minimize host crossings — batch a ptr+len over N items in one call, not N calls; pass pointers not values. wasm-bindgen/serde-wasm-bindgen marshal (know what they emit); instantiateStreaming + cache the compiled module, instantiate per-request. Linear memory only grows (no shrink) → bump/arena allocators for phase-scoped work.",
        triggers=(
            "wasm",
            "wasm-bindgen",
            "wasm_bindgen",
            "wasm32",
            "js-sys",
            "js_sys",
            "web-sys",
            "web_sys",
            "instantiate",
        ),
    ),
    Rule(
        "WASM build: wasm-opt -Oz/-O3 (Binaryen) as a mandatory second pass; strip debug+producers, lto=fat, codegen-units=1, panic=abort; twiggy to find size. SIMD (std::arch::wasm32) for hot vectorizable/crypto kernels (2–5x); wasm-feature-detect + two builds when SIMD/threads aren't universal.",
        triggers=(
            "wasm",
            "wasm-opt",
            "binaryen",
            "twiggy",
            "wasm-pack",
            "simd",
            "wasm32",
        ),
    ),
    Rule(
        "WASM off-browser: WASI + the Component Model (WIT interfaces via wit-bindgen) for language-neutral, capability-scoped plugins/verifiers; the sandbox = no ambient authority (no DOM/fs/net/clock unless imported); constant-time crypto is still your job (data-independent branches & memory access).",
        triggers=(
            "wasi",
            "component model",
            "wit-bindgen",
            "wasmtime",
            "wasmer",
            "plugin",
            "verifier",
            "wasm",
        ),
    ),
]
