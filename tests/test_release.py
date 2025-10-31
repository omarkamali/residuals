import textwrap
from pathlib import Path

from scripts.publish import extract_latest_changelog, extract_latest_changelog_from_file, SECTION_HEADER_RE


def test_section_header_regex_matches():
    assert SECTION_HEADER_RE.match("## [1.2.3] - 2025-10-31")
    assert SECTION_HEADER_RE.match("## [0.0.1] - 1999-01-01")
    assert not SECTION_HEADER_RE.match("## 1.2.3 - 2025-10-31")


def test_extract_latest_changelog_simple():
    txt = textwrap.dedent(
        """
        # Changelog
        ## [1.2.3] - 2025-10-31
        
        ### Added
        - A
        
        ## [1.2.2] - 2025-10-30
        
        ### Fixed
        - B
        """
    ).strip()
    version, section = extract_latest_changelog(txt)
    assert version == "1.2.3"
    assert "### Added" in section
    assert "- A" in section
    assert "1.2.2" not in section


def test_extract_latest_changelog_from_file(tmp_path: Path):
    content = textwrap.dedent(
        """
        # Changelog
        ## [9.9.9] - 2025-12-31
        
        ### Notes
        - Something
        
        ## [9.9.8] - 2025-12-30
        - Older
        """
    ).strip()
    p = tmp_path / "CHANGELOG.md"
    p.write_text(content, encoding="utf-8")
    version, section = extract_latest_changelog_from_file(p)
    assert version == "9.9.9"
    assert "Something" in section
