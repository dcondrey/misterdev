from misterdev.core.context.contracts.csharp_tree_sitter import (
    _extract_csharp_symbols_ts,
)

_SRC = """namespace App {
  public class Service : IService {
    public string Query(int x) => "";
    private int n;
    public int N { get; set; }
  }
  public interface IService { string Query(int x); }
  public record User(string Name, int Age);
  public enum Status { Ok, Failed }
  public struct Point { public int X; public int Y; }
}"""


def _syms():
    syms = _extract_csharp_symbols_ts(_SRC)
    if not syms:
        import pytest

        pytest.skip("csharp grammar unavailable")
    return syms


def test_public_class_and_method():
    syms = _syms()
    by_name = {s["name"]: s for s in syms}
    assert "Service" in by_name
    assert by_name["Service"]["kind"] == "class"
    assert "Service.Query" in by_name
    assert by_name["Service.Query"]["kind"] == "method"
    assert "Query" in by_name["Service.Query"]["signature"]


def test_interface():
    syms = _syms()
    iface = next(s for s in syms if s["name"] == "IService")
    assert iface["kind"] == "interface"
    # Interface members are implicitly public and surface as Type.Member.
    assert any(s["name"] == "IService.Query" for s in syms)


def test_record():
    syms = _syms()
    rec = next(s for s in syms if s["name"] == "User")
    assert rec["kind"] == "record"
    assert "Name" in rec["signature"] and "Age" in rec["signature"]


def test_enum_with_members():
    syms = _syms()
    en = next(s for s in syms if s["name"] == "Status")
    assert en["kind"] == "enum"
    assert "Ok" in en["signature"] and "Failed" in en["signature"]


def test_private_field_excluded():
    syms = _syms()
    names = {s["name"] for s in syms}
    assert "Service.n" not in names
    service = next(s for s in syms if s["name"] == "Service")
    assert " n" not in service["signature"]


def test_empty_and_garbage():
    assert _extract_csharp_symbols_ts("") == []
    assert _extract_csharp_symbols_ts("$$$ not valid <<< c#") == []
