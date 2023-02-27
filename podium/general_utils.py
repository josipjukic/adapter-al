import textwrap
from typing import Any, Dict


def repr_type_and_attrs(
    self: Any,
    attrs: Dict[str, Any],
    with_newlines: bool = False,
    repr_values: bool = True,
) -> str:
    delim = ",\n" if with_newlines else ", "
    attrs_str = delim.join(
        f"{k}: {repr(v) if repr_values else str(v)}" for k, v in attrs.items()
    )
    attrs_str = (
        f"\n{textwrap.indent(attrs_str, ' ' * 4)}\n" if with_newlines else attrs_str
    )
    return f"{type(self).__name__}({{{attrs_str}}})"
