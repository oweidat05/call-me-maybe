"""Core pipeline: reads prompts, runs constrained decoding, writes results."""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel


class ParameterDef(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef]


class FunctionCallResult(BaseModel):
    prompt: str
    name: Optional[str]
    parameters: Dict[str, Any]
    reason: Optional[str] = None


NUMBER_PREFIX_RE = re.compile(r"^-?\d*\.?\d*$")
NUMBER_CHARS = set("0123456789.-")

NO_MATCH_LOG_PROB_THRESHOLD = -0.05


def tokenize(model: Small_LLM_Model, text: str) -> List[int]:
    """Encode a piece of text into a list of token ids.

    Args:
        model: The loaded LLM wrapper used for tokenization.
        text: The raw text to encode.

    Returns:
        List[int]: The token ids produced by the model's tokenizer.
    """
    return cast(List[int], model.encode(text)[0].tolist())


def get_logits(model: Small_LLM_Model, input_ids: List[int]) -> List[float]:
    """Fetch the next-token logits for a given sequence of input ids.

    Args:
        model: The loaded LLM wrapper.
        input_ids: The full token id sequence generated so far.

    Returns:
        List[float]: The logit for every token in the vocabulary.
    """
    return cast(List[float], model.get_logits_from_input_ids(input_ids))


def encode_single_token(model: Small_LLM_Model, literal: str) -> int:
    """Encode a literal string that is expected to map to one token.

    Args:
        model: The loaded LLM wrapper.
        literal: The literal text expected to encode to a single token
            (e.g. a stop character such as ``","`` or ``"}"``).

    Returns:
        int: The single token id for the literal.

    Raises:
        AssertionError: If the literal encodes to more or less than
            exactly one token.
    """
    tokens = model.encode(literal)[0].tolist()
    if len(tokens) != 1:
        raise AssertionError(
            f"expected {literal!r} to encode to a single token, got {tokens}"
        )
    return cast(int, tokens[0])


def add_tokens(
    model: Small_LLM_Model,
    input_ids: List[int],
    logits: List[float],
    literal: str,
) -> List[float]:
    """Force-append the tokens of a fixed literal string to input_ids.

    This is used to inject JSON syntax (braces, quotes, keys) that must
    always appear verbatim, without letting the model choose it.

    Args:
        model: The loaded LLM wrapper.
        input_ids: The token id sequence to extend, mutated in place.
        logits: The logits from the previous decoding step (unused for
            selection here, kept for a consistent call signature).
        literal: The literal text to force-append.

    Returns:
        List[float]: The logits after appending every token of the
        literal, ready for the next constrained-decoding step.
    """
    for token_id in model.encode(literal)[0].tolist():
        input_ids.append(token_id)
        logits = model.get_logits_from_input_ids(input_ids)
    return logits


def add_tokens_batch(
    model: Small_LLM_Model, input_ids: List[int], literals: List[str]
) -> List[float]:
    """Encode and append a batch of string literals to input_ids, then return logits.

    Args:
        model: The small LLM model used for encoding and logits computation.
        input_ids: List of token ids to extend in place with encoded literals.
        literals: List of string literals to encode and append as tokens.

    Returns:
        Logits computed from the model given the updated input_ids.
    """

    for lit in literals:
        tokens = model.encode(lit)[0].tolist()
        input_ids.extend(tokens)
    return cast(List[float], model.get_logits_from_input_ids(input_ids))


def log_softmax(logits: List[float]) -> List[float]:
    """Compute the log-softmax of a list of logits.

    Args:
        logits: Raw logits for every token in the vocabulary.

    Returns:
        List[float]: The log-probability of each token.
    """
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    log_denom = math.log(sum(exps))
    return [(logit - max_logit) - log_denom for logit in logits]


def score_function_candidates(
    model: Small_LLM_Model,
    dic_name_token: Dict[str, List[int]],
    input_ids: List[int],
    base_logits: List[float],
) -> Dict[str, float]:
    """Score each candidate literal (function name or boolean) by its
    average per-token log-probability along the forced token path.

    Args:
        model: The loaded LLM wrapper.
        dic_name_token: Mapping from candidate name to its token ids.
        input_ids: The token id sequence right before the candidate
            starts.
        base_logits: The logits at that same position.
        {
            "fn_add": [100, 25, 40],
            "fn_greet": [100, 73, 90]
        }
        scores = {
           "fn_add_numbers": -1.2,
            "fn_greet": -3.5
            }

    Returns:
         Dict[str, float]: Mapping from candidate name to its average
        log-probability score. Higher (closer to zero) means the model
        is more confident in that candidate.
    """
    scores = {}
    for name, tokens in dic_name_token.items():
        path_ids = list(input_ids)
        logits = base_logits
        total = 0.0
        for i in range(1, len(tokens)):
            log_probs = log_softmax(logits)
            total += log_probs[tokens[i]]
            path_ids.append(tokens[i])
            logits = get_logits(model, path_ids)
        scores[name] = total / (len(tokens) - 1) if len(tokens) > 1 else total
    return scores


def build_number_token_map(vocab: Dict[str, int]) -> Dict[int, str]:
    """
    # "3234a" False

    Build a map of tokens that can be used to create numbers.

    The function checks every token in the vocabulary:

        Vocabulary
             |
             v
        Check the token
             |
        +----+----+
        |         |
      valid    invalid
        |         |
        v         v
       keep     ignore

    A token is kept if it contains only:
        digits (0-9), '.', or '-'

    Example:

        "345"   -> valid   -> keep
        ".5"    -> valid   -> keep
        "-12"   -> valid   -> keep
        "hello" -> invalid -> ignore

    The result is stored as:
        token ID -> token text

    Example:

        {
            100: "345",
            101: ".5",
            102: "-12"
        }
        token_str = "hello"
        token_id  = 100

    Args:
        vocab: Mapping from raw token string to token id, as loaded
            from the model's vocabulary file.

    Returns:
        Dict[int, str]: Mapping from token id to its cleaned string
        form, for every token made up only of digits, ``.`` or ``-``.
    """
    number_tokens = {}
    for token_str, token_id in vocab.items():
        stripped = token_str.lstrip("\u0120")
        if stripped and all(ch in NUMBER_CHARS for ch in stripped):
            number_tokens[token_id] = stripped
    return number_tokens


def generate_number(
    model: Small_LLM_Model,
    input_ids: List[int],
    logits: List[float],
    number_tokens: Dict[int, str],
    stop_token_id: int,
    max_digits: int = 12,
) -> Tuple[str, List[float]]:
    """Generate a numeric literal token-by-token under constrained decoding.

    At every step, only tokens that keep the accumulated text a valid
    number prefix (or the stop token) are considered.


    Generate a valid JSON number step by step.

    The function checks each possible token before adding it:

        current value + new token
                    |
                    v
             valid number prefix?
                /           \
              yes            no
               |              |
             keep           reject

    Example:

        value = "26"
        token = "5"
            |
            v
        "26" + "5" = "265"   -> valid -> keep

        value = "26"
        token = "-"
            |
            v
        "26" + "-" = "26-"   -> invalid -> reject

    The model can also choose a stop token when the number is complete.



    Args:
        model: The loaded LLM wrapper.
        input_ids: The token id sequence to extend, mutated in place.
        logits: The logits at the current decoding position.
        number_tokens: Mapping from token id to numeric-only string,
            as produced by :func:`build_number_token_map`.
        stop_token_id: The token id that ends this value (``","`` or
            ``"}"``, depending on parameter position).
        max_digits: Safety cap on how many tokens to generate.

    Returns:
        Tuple[str, List[float]]: The generated numeric literal as text,
        and the logits after generation.

    Raises:
        ValueError: If the model produced an empty or invalid literal
            (e.g. just ``"-"`` or ``"."``).
    """
    value = ""
    for _ in range(max_digits):
        candidates = {}
        for token_id, token_str in number_tokens.items():
            if NUMBER_PREFIX_RE.match(value + token_str):
                candidates[token_id] = token_str
        candidates[stop_token_id] = "<stop>"
        best_token = max(candidates, key=lambda token_id: logits[token_id])
        if best_token == stop_token_id:
            break
        value += candidates[best_token]
        input_ids.append(best_token)
        logits = model.get_logits_from_input_ids(input_ids)

    if not value or value in ("-", ".", "-."):
        raise ValueError(f"Model produced an invalid number literal: {value!r}")

    return value, logits


def generate_string(
    model: Small_LLM_Model,
    input_ids: List[int],
    logits: List[float],
    id_to_token: Dict[int, str],
    max_tokens: int = 80,
    max_chars: int = 500,
) -> Tuple[str, List[float]]:
    """Generate a JSON string value token-by-token, honoring JSON escaping.

    Args:
        model: The loaded LLM wrapper.
        input_ids: The token id sequence to extend, mutated in place.
        logits: The logits at the current decoding position.
        id_to_token: Reverse vocabulary mapping (token id -> string).
        max_tokens: Safety cap on how many tokens to generate.
        max_chars: Safety cap on the raw character length.

    Returns:
        Tuple[str, List[float]]: The decoded (unescaped) string value,
        and the logits after generation.
    """
    raw_chars: List[str] = []
    escaped = False
    is_first_token = True

    for _ in range(max_tokens):
        best_token = max(range(len(logits)), key=lambda i: logits[i])
        token_str = id_to_token.get(best_token, "").replace("\u0120", " ")

        if is_first_token and token_str.startswith(" "):
            token_str = token_str[1:]

        is_first_token = False

        finished = False
        for ch in token_str:
            if escaped:
                raw_chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                raw_chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                finished = True
                break
            raw_chars.append(ch)

        if finished:
            break

        input_ids.append(best_token)
        logits = model.get_logits_from_input_ids(input_ids)

        if sum(len(c) for c in raw_chars) > max_chars:
            break

    value = json.loads('"' + "".join(raw_chars) + '"')
    return value, logits


def build_context_prefix(function_definitions: Dict[str, Any]) -> str:
    """Build the system-prompt text listing every available function.

    Args:
        function_definitions: Mapping from function name to its full
            definition (description, parameters, etc.).
            [
                "You are a function-calling assistant. Available functions:",
                "- fn_add_numbers(a: number, b: number): Add two numbers",
                "- fn_greet(name: string): Greet someone",
            ]

    Returns:
        str: The system-prompt text to prepend to every request.
    """
    lines = ["You are a function-calling assistant. Available functions:"]
    for name, definition in function_definitions.items():
        params = ", ".join(
            f"{param_name}: {param_info['type']}"
            for param_name, param_info in definition["parameters"].items()
        )
        lines.append(f"- {name}({params}): {definition['description']}")
    lines.append(
        "Respond with a single JSON object in the form "
        '{"name": <function name>, "parameters": {...}}, '
        "calling exactly one of the functions above. No other text."
    )
    return "\n".join(lines)


def _load_json_file(path: str, label: str) -> Any:
    """Load a JSON file, raising a clear, catchable error on failure.

    Args:
        path: Filesystem path to the JSON file.
        label: Human-readable name of the file, used in error messages
            (e.g. ``"Input test file"``).

    Returns:
        Any: The parsed JSON content
        read file and return from json to the python object.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ValueError: If the file exists but is not valid JSON.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at path: {path!r}")
    with open(path, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} at {path!r} is not valid JSON: {error}"
            ) from error


def _process_prompt(
    model: Small_LLM_Model,
    prompt: str,
    context_prefix: str,
    dic_name_token: Dict[str, List[int]],
    function_definitions: Dict[str, Any],
    first_token_id: int,
    number_tokens: Dict[int, str],
    id_to_token: Dict[int, str],
) -> Dict[str, Any]:
    """Run the full constrained-decoding pipeline for a single prompt.

    Args:
        model: The loaded LLM wrapper.
        prompt: The natural-language prompt to translate into a call.
        context_prefix: The system-prompt text listing all functions.
        dic_name_token: Mapping from function name to its token ids.
        function_definitions: Mapping from function name to its full
            definition (parameters and their types).
        first_token_id: The token id shared by every function name's
            first token, used to seed the name-selection step.
        number_tokens: Mapping from token id to numeric-only string.
        id_to_token: Reverse vocabulary mapping (token id -> string).

        <|im_start|>system
        You are a function-calling assistant. Available functions:
            - fn_add_numbers(a: number, b: number): Add two numbers
            ...
        <|im_end|>
        <|im_start|>user
            What is the sum of 2 and 3?
        <|im_end|>
        <|im_start|>assistant
        <think>
        </think>

    Returns:
        Dict[str, Any]: A single result entry with ``prompt``, ``name``,
        and ``parameters`` (or ``name: None`` with a ``reason`` if no
        function matched confidently).

    Raises:
        ValueError: If a parameter's declared type is not one of
            ``integer``, ``number``, ``boolean``, or ``string``.
    """

    full_input_text = (
        "<|im_start|>system\n" + context_prefix + "<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n"
    )

    input_ids = tokenize(model, full_input_text)
    logits = get_logits(model, input_ids)

    logits = add_tokens_batch(model, input_ids, ["{", '"', "name", '"', ":", '"'])

    input_ids.append(first_token_id)
    logits = get_logits(model, input_ids)

    scores = score_function_candidates(model, dic_name_token, input_ids, logits)
    selected_function = max(scores, key=lambda name: scores[name])
    best_score = scores[selected_function]

    if best_score < NO_MATCH_LOG_PROB_THRESHOLD:
        return {
            "prompt": prompt,
            "name": None,
            "parameters": {},
        }

    for token_id in dic_name_token[selected_function][1:]:
        input_ids.append(token_id)
    logits = get_logits(model, input_ids)

    logits = add_tokens_batch(model, input_ids, ['"', ",", '"parameters"', ":", "{"])

    parameters = function_definitions[selected_function]["parameters"]
    parameter_names = list(parameters.keys())
    result_parameters: Dict[str, Any] = {}

    for index, parameter_name in enumerate(parameter_names):
        parameter_type = parameters[parameter_name]["type"]
        is_last = index == len(parameter_names) - 1
        stop_literal = "}" if is_last else ","
        stop_token_id = encode_single_token(model, stop_literal)

        logits = add_tokens_batch(model, input_ids, ['"', parameter_name, '"', ":"])

        if parameter_type == "integer":
            value, logits = generate_number(
                model, input_ids, logits, number_tokens, stop_token_id
            )
            result_parameters[parameter_name] = int(float(value))
            logits = add_tokens(model, input_ids, logits, stop_literal)

        elif parameter_type == "number":
            value, logits = generate_number(
                model, input_ids, logits, number_tokens, stop_token_id
            )
            result_parameters[parameter_name] = float(value)
            logits = add_tokens(model, input_ids, logits, stop_literal)

        elif parameter_type == "boolean":
            bool_candidates = {
                "true": tokenize(model, "true"),
                "false": tokenize(model, "false"),
            }
            bool_scores = score_function_candidates(
                model, bool_candidates, input_ids, logits
            )
            chosen = max(bool_scores, key=lambda name: bool_scores[name])

            for token_id in bool_candidates[chosen][1:]:
                input_ids.append(token_id)
            logits = get_logits(model, input_ids)
            result_parameters[parameter_name] = chosen == "true"
            logits = add_tokens(model, input_ids, logits, stop_literal)

        elif parameter_type == "string":
            logits = add_tokens(model, input_ids, logits, '"')
            value, logits = generate_string(model, input_ids, logits, id_to_token)

            if (
                value
                and value[-1] in ('"', "'")
                and value[:-1] in prompt
                and value not in prompt
            ):
                value = value[:-1]
            result_parameters[parameter_name] = value
            logits = add_tokens(model, input_ids, logits, '"')
            logits = add_tokens(model, input_ids, logits, stop_literal)

        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type!r}")

    result = FunctionCallResult(
        prompt=prompt, name=selected_function, parameters=result_parameters
    )
    return cast(Dict[str, Any], result.model_dump(exclude_none=True))


def run_pipeline(
    functions_definition_path: str, input_path: str, output_path: str
) -> None:
    """Run the full function-calling pipeline and write results to disk.

    Args:
        functions_definition_path: Path to the functions definition JSON.
        input_path: Path to the JSON array of prompts to process.
        output_path: Path where the resulting JSON array is written.


    Raises:
        FileNotFoundError: If an input file does not exist.
        ValueError: If an input file is not valid JSON, or if function
            names don't share a common first token.
    """
    tests = _load_json_file(input_path, "Input test file")
    functions = _load_json_file(functions_definition_path, "Functions definition file")
    raw_functions = _load_json_file(
        functions_definition_path, "Functions definition file"
    )

    functions = []
    for func in raw_functions:
        validated_func = FunctionDef(**func)
        functions.append(validated_func.model_dump())
    function_names = [function["name"] for function in functions]
    # loop for all the name function
    function_definitions = {function["name"]: function for function in functions}
    # {
    # fn_add:{
    # name : fn_add
    # discription: add number
    # parameters: {a: {type: integer}}
    # }
    model = Small_LLM_Model()

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as file:
        vocab = json.load(file)
    id_to_token = {token_id: token for token, token_id in vocab.items()}
    number_tokens = build_number_token_map(vocab)

    dic_name_token = {name: tokenize(model, name) for name in function_names}
    first_tokens = {tokens[0] for tokens in dic_name_token.values()}
    if len(first_tokens) != 1:
        raise ValueError("function names don't share one common first token")
    first_token_id = next(iter(first_tokens))

    context_prefix = build_context_prefix(function_definitions)

    results: List[Dict[str, Any]] = []

    for test in tests:
        prompt: Optional[str] = test.get("prompt")

        if not prompt or not str(prompt).strip():
            print(f"Skipping malformed test entry (empty or missing prompt): {test}")
            results.append(
                {
                    "prompt": prompt if prompt is not None else "",
                    "name": None,
                    "parameters": {},
                    "reason": "empty or missing prompt",
                }
            )
            continue

        try:
            result = _process_prompt(
                model,
                prompt,
                context_prefix,
                dic_name_token,
                function_definitions,
                first_token_id,
                number_tokens,
                id_to_token,
            )
            print("result:", result)
            results.append(result)

        except Exception as error:
            print(f"Error processing prompt {prompt!r}: {error}")
            results.append(
                {
                    "prompt": prompt,
                    "name": None,
                    "parameters": {},
                    "reason": f"error: {error}",
                }
            )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as out_file:
        json.dump(results, out_file, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(results)} / {len(tests)} prompts")
    print(f"Results written to: {output_path}")
