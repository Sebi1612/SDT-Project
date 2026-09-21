"""Build the pinned Tree-sitter language libraries used by the analyses.

The project uses tree_sitter 0.20, which accepts grammar ABI 13 or 14. The
checked-in generated parsers are pinned to ABI 14. Building explicitly with
the system C/C++ compilers avoids a setuptools linker regression in
Language.build_library when a grammar has a C++ scanner.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
GRAMMARS = {
    # The original paper's grammar is ABI 14. The newer root grammar is kept
    # as an upstream source snapshot but emits ABI 15, which tree_sitter 0.20
    # cannot load.
    "python": ("attention/tree-sitter-python", ("parser.c", "scanner.cc")),
    "java": ("tree-sitter-java", ("parser.c",)),
    "go": ("tree-sitter-go", ("parser.c",)),
    "javascript": ("tree-sitter-javascript", ("parser.c", "scanner.c")),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build(language: str, output_dir: Path, cc: str, cxx: str) -> Path:
    grammar_name, source_names = GRAMMARS[language]
    source_dir = REPO / grammar_name / "src"
    sources = [source_dir / name for name in source_names]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing parser sources: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"my-languages-{language}.so"

    if any(path.suffix in {".cc", ".cpp", ".cxx"} for path in sources):
        with tempfile.TemporaryDirectory(prefix=f"tree-sitter-{language}-") as tmp:
            objects = []
            for source in sources:
                compiler = cxx if source.suffix in {".cc", ".cpp", ".cxx"} else cc
                obj = Path(tmp) / f"{source.stem}.o"
                run([
                    compiler, "-fPIC", "-I", str(source_dir), "-c",
                    str(source), "-o", str(obj),
                ])
                objects.append(str(obj))
            run([cxx, "-shared", *objects, "-o", str(output)])
    else:
        run([
            cc, "-fPIC", "-shared", "-I", str(source_dir),
            *(str(source) for source in sources), "-o", str(output),
        ])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", choices=GRAMMARS, default=list(GRAMMARS)
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "build")
    parser.add_argument("--cc", default=shutil.which("gcc") or "cc")
    parser.add_argument("--cxx", default=shutil.which("g++") or "c++")
    args = parser.parse_args()
    for language in args.languages:
        output = build(language, args.output_dir, args.cc, args.cxx)
        print(f"built {language}: {output}")


if __name__ == "__main__":
    main()
