from __future__ import annotations

from pathlib import Path


def path_scratch(temp_file_suffix: bool = True) -> str:
    if temp_file_suffix:
        return (Path(__file__).parent.parent / ".scratch" / "temp_files").resolve()
    else:
        return (Path(__file__).parent.parent / ".scratch").resolve()


# print(path_scratch())
