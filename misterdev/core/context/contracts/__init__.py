"""Cross-task interface contract registry.

After each task completes, extracts the public API it created or modified.
Before executing downstream tasks, injects those contracts into prompts
so the LLM knows the exact signatures it must honor.

This addresses ~30% of multi-task build failures where one task assumes
a different interface than what the previous task actually created.
"""

from ._log import logger
from ._text import _extract_name
from .extraction import _extract_public_symbols
from .python_generic import _extract_generic_symbols, _extract_python_symbols
from .registry import Contract, ContractRegistry
from .rust_line import (
    _collect_enum_variants,
    _collect_signature,
    _collect_struct_fields,
    _collect_trait_methods,
    _extract_generics,
    _extract_impl_methods,
    _extract_impl_name,
    _extract_rust_symbols,
    _strip_visibility,
)
from .rust_tree_sitter import (
    _extract_rust_symbols_ts,
    _ts_decl,
    _ts_field_text,
    _ts_is_pub,
    _ts_pub_members,
    _ts_trait_methods,
    _ts_variant_names,
    _walk_rust_ts,
)

__all__ = [
    "logger",
    "Contract",
    "ContractRegistry",
    "_extract_public_symbols",
    "_extract_name",
    "_extract_generic_symbols",
    "_extract_python_symbols",
    "_extract_rust_symbols",
    "_extract_rust_symbols_ts",
    "_strip_visibility",
    "_extract_generics",
    "_extract_impl_name",
    "_extract_impl_methods",
    "_collect_enum_variants",
    "_collect_trait_methods",
    "_collect_signature",
    "_collect_struct_fields",
    "_ts_is_pub",
    "_ts_field_text",
    "_ts_decl",
    "_walk_rust_ts",
    "_ts_pub_members",
    "_ts_variant_names",
    "_ts_trait_methods",
]
