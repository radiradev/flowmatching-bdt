import pathlib

import pytest
from mktestdocs import check_md_file

DOCS_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs"


def _doc_files():
    for fpath in sorted(DOCS_DIR.glob("**/*.md")):
        yield fpath


@pytest.mark.parametrize("fpath", _doc_files(), ids=lambda p: str(p.relative_to(DOCS_DIR)))
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)
