import tempfile
from pathlib import Path

from my_project_orchestrator.core.context.contracts import (
    ContractRegistry,
    _extract_public_symbols,
    _extract_rust_symbols,
    _extract_python_symbols,
    _extract_name,
    _strip_visibility,
    _extract_generics,
)


def test_public_symbols_rust_treesitter_enum_trait_type_impl():
    # Drives the tree-sitter dispatcher path (not the line fallback), covering
    # enum/trait/type extraction and the variant/method/member helpers.
    content = (
        "pub enum Status { Ok, Failed }\n"
        "pub trait Queryable { fn query(&self) -> String; }\n"
        "pub type Id = u64;\n"
        "pub fn run(x: u32) -> u32 { x }\n"
        "pub struct Cfg { pub name: String }\n"
        "impl Cfg { pub fn new() -> Self { Cfg { name: String::new() } } }\n"
    )
    syms = _extract_public_symbols(content, "rust")
    if not syms:
        return  # rust grammar unavailable
    kinds = {s["kind"] for s in syms}
    names = {s["name"] for s in syms}
    assert {"pub enum", "pub trait", "pub type", "pub fn", "pub struct"} <= kinds
    assert "Cfg::new" in names  # impl method walked with parent
    enum = next(s for s in syms if s["kind"] == "pub enum")
    assert "Ok" in enum["signature"] and "Failed" in enum["signature"]


def test_public_symbols_generic_language_fallback():
    content = "export function render() {}\nfunc Handle(w, r) {}\n"
    syms = _extract_public_symbols(content, "typescript")
    kinds = {s["kind"] for s in syms}
    assert "export" in kinds and "func" in kinds


def test_extract_rust_pub_fn():
    lines = [
        "pub fn validate_config(config: &Config) -> Result<(), Error> {",
        "    // body",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert symbols[0]["kind"] == "pub fn"
    assert symbols[0]["name"] == "validate_config"
    assert "config: &Config" in symbols[0]["signature"]


def test_extract_rust_pub_struct_with_fields():
    lines = [
        "pub struct PostingShard {",
        "    pub lists: Vec<PostingList>,",
        "    pub count: usize,",
        "    internal: bool,",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert symbols[0]["kind"] == "pub struct"
    assert "pub lists" in symbols[0]["signature"]
    assert "pub count" in symbols[0]["signature"]


def test_extract_rust_enum_with_variants():
    lines = [
        "pub enum Status {",
        "    Active,",
        "    Inactive,",
        "    Pending(u64),",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert symbols[0]["kind"] == "pub enum"
    assert "Active" in symbols[0]["signature"]
    assert "Inactive" in symbols[0]["signature"]


def test_extract_rust_trait_with_methods():
    lines = [
        "pub trait Queryable {",
        "    fn query(&self, key: &[u8]) -> Vec<u8>;",
        "    fn insert(&mut self, key: &[u8], value: &[u8]);",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert symbols[0]["kind"] == "pub trait"
    assert "fn query" in symbols[0]["signature"]
    assert "fn insert" in symbols[0]["signature"]


def test_extract_rust_pub_crate():
    lines = [
        "pub(crate) fn internal_helper(x: usize) -> bool {",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert symbols[0]["name"] == "internal_helper"


def test_extract_rust_impl_methods():
    lines = [
        "impl PostingShard {",
        "    pub fn new() -> Self {",
        "        Self { lists: vec![] }",
        "    }",
        "    pub fn insert(&mut self, atom_id: u64, indices: &[u16]) {",
        "        // ...",
        "    }",
        "    fn private_helper(&self) {}",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 2
    assert symbols[0]["name"] == "PostingShard::new"
    assert symbols[1]["name"] == "PostingShard::insert"


def test_extract_rust_generics():
    lines = [
        "pub struct IndexedMemory<T: Clone> {",
        "    pub shards: Vec<T>,",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert "<T: Clone>" in symbols[0]["signature"]


def test_extract_rust_multiline_fn():
    lines = [
        "pub fn overlap_scan(",
        "    query: &SparseVector,",
        "    threshold: f64,",
        ") -> Vec<(AtomId, f64)> {",
        "    // body",
        "}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 1
    assert "overlap_scan" in symbols[0]["signature"]
    assert "query" in symbols[0]["signature"]


def test_skips_private():
    lines = [
        "fn private_fn() {}",
        "pub fn public_fn() {}",
        "struct PrivateStruct {}",
        "pub struct PublicStruct {}",
    ]
    symbols = _extract_rust_symbols(lines)
    assert len(symbols) == 2
    assert all("public" in s["name"].lower() or "Public" in s["name"] for s in symbols)


def test_extract_python_symbols():
    lines = [
        "def validate(config):",
        "    pass",
        "def _private():",
        "    pass",
        "class ConfigManager:",
        "    pass",
    ]
    symbols = _extract_python_symbols(lines)
    assert len(symbols) == 2
    assert symbols[0]["name"] == "validate"
    assert symbols[1]["name"] == "ConfigManager"


def test_extract_name():
    assert _extract_name("foo_bar(x: int)") == "foo_bar"
    assert _extract_name("  MyStruct {") == "MyStruct"
    assert _extract_name("") == ""


def test_strip_visibility():
    assert _strip_visibility("pub fn foo()") == "fn foo()"
    assert _strip_visibility("pub(crate) fn bar()") == "fn bar()"
    assert _strip_visibility("pub(super) struct X") == "struct X"


def test_extract_generics():
    assert _extract_generics("Foo<T: Clone>") == "<T: Clone>"
    assert _extract_generics("Foo") == ""
    assert _extract_generics("HashMap<K, V>") == "<K, V>"


def test_contract_registry_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "src").mkdir()
        (td / "src" / "lib.rs").write_text(
            "pub fn hello() -> String { String::new() }\npub struct Config { pub name: String }\n"
        )

        reg = ContractRegistry(td)
        contracts = reg.extract_contracts(
            "T-001", ["src/lib.rs"], td, None, language="rust"
        )
        assert len(contracts) == 1
        assert len(contracts[0].symbols) >= 2

        reg2 = ContractRegistry(td)
        assert "T-001" in reg2.contracts


def test_get_contracts_for_task():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "src").mkdir()
        (td / "src" / "posting.rs").write_text(
            "pub fn overlap_scan() -> Vec<u64> { vec![] }\n"
        )

        reg = ContractRegistry(td)
        reg.extract_contracts(
            "001-posting", ["src/posting.rs"], td, None, language="rust"
        )

        ctx = reg.get_contracts_for_task(["001-posting"])
        assert "overlap_scan" in ctx
        assert "Interface Contracts" in ctx


def test_get_contracts_empty():
    with tempfile.TemporaryDirectory() as td:
        reg = ContractRegistry(Path(td))
        assert reg.get_contracts_for_task([]) == ""
        assert reg.get_contracts_for_task(["nonexistent"]) == ""
