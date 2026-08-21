import os
import re
import subprocess
import html
from pathlib import Path
from survey_kit import logger
from survey_kit import config
from survey_kit.utilities.inputs import create_folders_if_needed

from survey_kit.orchestration.from_decorator import as_function
from survey_kit.orchestration.callers import (
    UpdateParams,
    function_call_or_list,
    run_function_list,
)
from survey_kit.orchestration.utilities import CallInputs


def run_tutorials_in_path(
    path: str,
    sub_directories: bool = True,
    run_files: bool = True,
):
    """
    Run any tutorials in a given directory and recursively
    on each subdirectory
    """

    acceptable_suffixes = [".py"]

    path = os.path.normpath(path)
    if not os.path.exists(path):
        message = f"Path does not exist: {path}"
        logger.error(message)
        raise Exception(message)

    files_to_run = []
    for item_name in os.listdir(path):
        item_path = os.path.join(path, item_name)

        if os.path.isfile(item_path):
            valid_type = any(
                [item_path.endswith(suffixi) for suffixi in acceptable_suffixes]
            )

            if valid_type:
                files_to_run.append(item_path)
        elif os.path.isdir(item_path) and sub_directories:
            # Recursively call the function for subdirectories
            files_to_run.extend(
                run_tutorials_in_path(
                    path=item_path, sub_directories=sub_directories, run_files=False
                )
            )

    if run_files:
        for filei in files_to_run:
            f = run_jupyter_to_html.as_function(path=Path(filei).as_posix())

            function_call_or_list(
                [f],
                run=False,
                return_ordering=True,
                update=UpdateParams(update_by_date=True, update_by_used_file_list=True),
                show_only_functions_set_to_run=False,
            )

            if f.run:
                # run_jupyter_to_html(filei)
                run_function_list(f, run_all=True)
            else:
                logger.info(
                    f"Not running '{os.path.basename(filei)}' as it is already up to date."
                )
    else:
        return files_to_run


def path_to_html(path: str = "") -> str:
    return Path(path).with_suffix(".html").as_posix()


@as_function(
    outputs=path_to_html,
    args_outputs=["path"],
    inputs_parameters=["path"],
    call_input=CallInputs(n_cpu=config.cpus),
)
def run_jupyter_to_html(path: str):
    """
    Runs a jupyter notebook from a .py file
    and saves the output to an HTML file, compatible with Windows and Linux.

    Args:
        notebook_path: The file path to the input .py marimo notebook.
        output_path: The file path where the output .html file should be saved.
    """

    path_dir = os.path.dirname(path) or os.getcwd()
    path_name = os.path.basename(path)
    output_path = path_to_html(path)
    output_filename = os.path.basename(output_path)

    output_full_path = (Path(path_dir) / Path(output_filename)).as_posix()
    if os.path.exists(output_full_path):
        os.remove(output_full_path)

    # Define the command and arguments as a list for cross-platform safety
    command_convert_args = ["uv", "run", "jupytext", "--to", "ipynb", path_name]

    command_convert = [
        "uv",
        "run",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--execute",
        Path(path_name).with_suffix(".ipynb").as_posix(),
    ]

    print(f"Executing command: {' '.join(command_convert_args)}")
    print(f"In directory: {path_dir}")

    try:
        # Run the command, setting the current working directory (cwd)
        # This is safer and more robust than trying to use `cd` in the command string
        result = subprocess.run(
            command_convert_args,
            cwd=path_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        result_final = subprocess.run(
            command_convert, cwd=path_dir, check=True, capture_output=True, text=True
        )
        logger.info(f"\nSuccess! file '{path_name}' successfully exported to html.")
        os.remove((Path(path_dir) / Path(path_name).with_suffix(".ipynb")).as_posix())

        import shutil

        output_docs = (
            Path(output_path).as_posix().replace("/tutorials", "/docs/tutorials")
        )
        create_folders_if_needed(os.path.dirname(output_docs))
        if os.path.isfile(output_docs):
            os.remove(output_docs)

        shutil.copy(output_path, output_docs)
        # escape_html_code(output_path)

    except subprocess.CalledProcessError as e:
        logger.error(
            f"\nAn error occurred during execution (Error Code {e.returncode}):"
        )
        logger.error("STDOUT:", e.stdout)
        logger.error("STDERR:", e.stderr)
    except FileNotFoundError:
        logger.error("\nError: The 'uv' or 'jupyter' command was not found.")
        logger.error(
            "Please ensure uv is installed and accessible in your system's PATH."
        )


#   logger.info(".[!n]") / logger.info(f"{n}[!n]") (see replicates.py) rely on a
#   no-newline terminator that works in a real terminal, but nbconvert --execute
#   emits each logging call as its own <pre> output block, so every "." and
#   replicate number ends up on its own line. Collapse consecutive dot/number
#   blocks back onto one line. The blank-line block emitted every 50 replicates
#   (replicate_number % 50 == 0) is absorbed into the same merged line as a real
#   "\n" rather than left as a separate block, and a "\n" is force-inserted every
#   PROGRESS_BREAK_EVERY replicates even when no such marker was emitted (e.g. a
#   survey with fewer than 50 replicate weights), so line breaks land at
#   predictable, replicate-count-based points instead of wherever pre-wrap CSS
#   happens to wrap the raw text.
PROGRESS_BREAK_EVERY = 50
_OUTPUT_CHILD_RE = re.compile(
    r'<div class="jp-OutputArea-child">\s*'
    r'<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>\s*'
    r'<div class="jp-RenderedText jp-OutputArea-output"([^>]*)>\s*'
    r"<pre>([^<]*)</pre>\s*"
    r"</div>\s*"
    r"</div>"
)
_PROGRESS_TOKEN_RE = re.compile(r"^(?:\.|\d+)$")
_PROGRESS_BREAK_TEXT = "\n"


def merge_progress_dot_outputs(content: str) -> str:
    matches = list(_OUTPUT_CHILD_RE.finditer(content))

    result = []
    pos = 0
    i = 0
    while i < len(matches):
        m = matches[i]
        if not _PROGRESS_TOKEN_RE.match(m.group(2)):
            i += 1
            continue

        run_items = [m.group(2)]
        run_end = i
        j = i + 1
        while j < len(matches):
            gap = content[matches[j - 1].end() : matches[j].start()]
            if gap.strip() != "":
                break
            text = matches[j].group(2)
            if not (_PROGRESS_TOKEN_RE.match(text) or text == _PROGRESS_BREAK_TEXT):
                break
            run_items.append(text)
            run_end = j
            j += 1

        if run_end == i:
            i += 1
            continue

        result.append(content[pos : m.start()])
        result.append(
            '<div class="jp-OutputArea-child">\n'
            '<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>\n'
            f'<div class="jp-RenderedText jp-OutputArea-output"{m.group(1)}>\n'
            f"<pre>{_render_progress_run(run_items)}</pre>\n"
            "</div>\n"
            "</div>"
        )
        pos = matches[run_end].end()
        i = run_end + 1

    result.append(content[pos:])
    return "".join(result)


#   How close a forced break is allowed to land next to a natural (source-emitted)
#   one before we skip forcing and just wait the extra few tokens for the real
#   one - avoids stranding a single token (e.g. "50") on its own line between a
#   forced break and the natural one that follows almost immediately after.
_PROGRESS_BREAK_LOOKAHEAD = 5


def _render_progress_run(run_items: list) -> str:
    break_positions = [
        idx for idx, text in enumerate(run_items) if text == _PROGRESS_BREAK_TEXT
    ]

    parts = []
    count_since_break = 0
    for idx, text in enumerate(run_items):
        if text == _PROGRESS_BREAK_TEXT:
            if parts and parts[-1] != _PROGRESS_BREAK_TEXT:
                parts.append(_PROGRESS_BREAK_TEXT)
            count_since_break = 0
            continue

        parts.append(text)
        count_since_break += 1

        if count_since_break >= PROGRESS_BREAK_EVERY:
            upcoming_break = next((p for p in break_positions if p > idx), None)
            if (
                upcoming_break is not None
                and upcoming_break - idx <= _PROGRESS_BREAK_LOOKAHEAD
            ):
                continue
            parts.append(_PROGRESS_BREAK_TEXT)
            count_since_break = 0

    while parts and parts[-1] == _PROGRESS_BREAK_TEXT:
        parts.pop()

    return "".join(parts)


def merge_progress_dot_outputs_in_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = merge_progress_dot_outputs(content)
    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)


def escape_html_code(path: str):
    # Read the original, rendered HTML content
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Escape the HTML tags (converts < to &lt; and > to &gt;)
    escaped_content = html.escape(content)

    # Write the escaped content to the new file
    with open(path, "w", encoding="utf-8") as f:
        f.write(escaped_content)


def run_all_tutorials():
    path = Path(config.code_root).parent.parent
    path_scratch = (path / ".scratch").as_posix()
    path_tutorials = (path / "tutorials").as_posix()

    config.data_root = path_scratch
    run_tutorials_in_path(path_tutorials)

    for html_dir in [Path(path_tutorials), path / "docs" / "tutorials"]:
        if html_dir.exists():
            for html_path in html_dir.rglob("*.html"):
                merge_progress_dot_outputs_in_file(html_path.as_posix())


if __name__ == "__main__":
    run_all_tutorials()
