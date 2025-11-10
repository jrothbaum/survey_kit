import os
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


if __name__ == "__main__":
    run_all_tutorials()
