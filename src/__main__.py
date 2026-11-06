"""Entry point for the function-calling tool.

Checks that required dependencies are installed before importing anything
else, so a missing package produces a clear message instead of a raw
traceback.
"""

import argparse
import sys

try:
    import numpy  # noqa: F401
    from pydantic import BaseModel  # noqa: F401

    from llm_sdk.llm_sdk import Small_LLM_Model  # noqa: F401
except ImportError as error:
    missing = getattr(error, "name", None) or str(error)
    print(
        f"Missing required dependency: {missing}\n"
        "Please install project dependencies before running this program, "
        "e.g. run 'uv sync' (or 'make install')."
    )
    sys.exit(1)

from src.core import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the function-calling tool.

    Returns:
        argparse.Namespace: Parsed arguments with paths to the function
        definitions file, the input test file, and the output file.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to write the output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full function-calling pipeline from CLI arguments."""
    args = parse_args()
    try:
        run_pipeline(
            functions_definition_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError) as error:
        # Missing files or malformed JSON in the *input* files themselves.
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
