"""Locate the shipped demo tree, from wherever a notebook happens to be running.

**Private on purpose.** The leading underscore keeps this out of the 0.9.0 public API
(same convention as :mod:`electropycal._parallel`): it is scaffolding so the shipped
notebooks have something to open, not a path API anyone should build on. It lives
*inside the package* rather than next to the notebooks because in Colab the notebook is
uploaded on its own and the pip-installed wheel is the only thing present; a sibling
``notebooks/_demo.py`` would not be importable there.

The problem it solves: every public notebook used to hardcode ``ROOT =
"demo/in_vitro/input"``, a path relative to the repo root. A Jupyter kernel's working
directory is the *notebook's own* directory, so opening ``notebooks/x.ipynb`` and
running it gave ``FileNotFoundError: 'demo\\in_vitro\\input'`` several cells in. Swapping
in ``"../demo/..."`` just moves the breakage: it would then work from ``notebooks/`` and
fail from the repo root, from VS Code with a workspace-root cwd, and in Colab.

So nothing here counts parent directories. Candidate roots are probed in order and each
is accepted only if the tree it should contain is really there:

1. ``$ELECTROPYCAL_DEMO``: explicit override, for CI or an unpacked release.
2. the current working directory and each of its ancestors: covers the repo root,
   ``notebooks/``, ``publish/``, and anywhere else inside a checkout.
3. the installed package's own location: covers an editable/src-layout install driven
   from an unrelated cwd. A wheel install has no demo tree next to it, so this simply
   does not match, which is the correct answer rather than a wrong guess.

If none match (the ordinary Colab case, where there is no checkout at all),
:func:`demo_input` synthesizes an equivalent tree instead of failing.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["DemoTreeNotFound", "demo_input", "find_demo_tree", "require_dir"]

#: Demo kinds and the generator that can stand in for each when nothing is on disk.
_KINDS = {"in_vitro": "write_synthetic_pstrace_dir", "in_vivo": "write_synthetic_invivo_dir"}


class DemoTreeNotFound(FileNotFoundError):
    """Raised when the demo tree cannot be located and synthesis was not wanted."""


def _candidate_roots() -> list[Path]:
    """Roots that might contain a ``demo/`` directory, most explicit first."""
    roots: list[Path] = []

    env = os.environ.get("ELECTROPYCAL_DEMO")
    if env:
        p = Path(env).expanduser()
        # Accept either the tree root (containing demo/) or demo/ itself.
        roots += [p, p.parent]

    cwd = Path.cwd().resolve()
    roots += [cwd, *cwd.parents]

    # src-layout editable install: .../<root>/src/electropycal/_demo.py -> parents[2] is <root>.
    pkg = Path(__file__).resolve()
    roots += [pkg.parents[2], pkg.parents[1]] if len(pkg.parents) > 2 else []

    seen, uniq = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _relative_targets(kind: str) -> list[Path]:
    """Sub-paths, relative to a candidate root, that would BE the demo input tree."""
    return [Path("demo") / kind / "input", Path(kind) / "input"]


def find_demo_tree(kind: str = "in_vitro") -> Path | None:
    """Return the shipped demo *input* tree for ``kind``, or ``None`` if unreachable.

    ``None`` is a normal outcome, not an error: it means no checkout is present (Colab),
    and the caller should synthesize a tree instead.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown demo kind {kind!r}; expected one of {sorted(_KINDS)}")
    for root in _candidate_roots():
        for rel in _relative_targets(kind):
            cand = root / rel
            if cand.is_dir() and any(cand.iterdir()):
                return cand.resolve()
    return None


def _searched_report(kind: str) -> str:
    """Human-readable account of what was looked for and where, never a bare errno."""
    rels = " or ".join(str(r) for r in _relative_targets(kind))
    lines = [f"could not find the shipped demo tree for kind={kind!r}.",
             f"  looked for : {rels}",
             f"  under      : {len(_candidate_roots())} candidate root(s):"]
    lines += [f"               {r}" for r in _candidate_roots()[:12]]
    lines += ["",
              "  Fixes, in order of least surprise:",
              "    - run scripts/build_demo_dataset.py --out demo   (builds it, ~15 s)",
              "    - set ELECTROPYCAL_DEMO=/path/to/the/tree/containing/demo",
              "    - point ROOT at your own PSTrace export directory instead"]
    return "\n".join(lines)


def require_dir(path, what: str = "directory") -> str:
    """Validate a caller-supplied path, reporting what was expected rather than an errno."""
    p = Path(path).expanduser()
    if p.is_dir():
        return str(p.resolve())
    raise DemoTreeNotFound(
        f"{what} not found: {p!r}\n"
        f"  resolved to : {p.resolve() if not p.is_absolute() else p}\n"
        f"  cwd         : {Path.cwd()}\n"
        f"  A relative path is resolved against the KERNEL's working directory, which for a\n"
        f"  notebook is the notebook's own folder -- not the repository root.")


def demo_input(kind: str = "in_vitro", user_path=None, *,
               synthesize: bool = True, **synth_kwargs) -> str:
    """One call that every public notebook can use for its input tree.

    ``user_path``  -- if given, validated and returned (clear error if absent).
    otherwise      -- the shipped demo tree if any candidate root has one,
    otherwise      -- a freshly synthesized equivalent, unless ``synthesize=False``.

    Returns a string path, because the review classes take ``str | Path`` and notebooks
    print it.
    """
    if user_path is not None:
        return require_dir(user_path, f"{kind} input directory")

    found = find_demo_tree(kind)
    if found is not None:
        return str(found)

    if not synthesize:
        raise DemoTreeNotFound(_searched_report(kind))

    from . import data  # noqa: F401  (ensure the subpackage is importable before getattr)
    from .data import synthetic
    writer = getattr(synthetic, _KINDS[kind])
    return str(writer(tempfile.mkdtemp(prefix=f"electropycal-demo-{kind}-"), **synth_kwargs))
