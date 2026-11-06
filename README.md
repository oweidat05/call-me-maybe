*This activity has been created as part of the 42 curriculum by mowaidat*

# call me maybe — Function Calling with Constrained Decoding

## Description

This project turns a plain natural-language question (e.g. *"What is the product of 3 and
5?"*) into a structured, machine-executable function call:

```json
{"prompt": "What is the product of 3 and 5?", "name": "fn_multiply_numbers", "parameters": {"a": 3, "b": 5}}
```

The core constraint of this project is that the output JSON is **never produced by asking
the model nicely and hoping it complies**. A small language model (`Qwen/Qwen3-0.6B`, ~0.6B
parameters) is, by itself, unreliable at producing valid structured output — it might succeed
30% of the time on a bare prompt. Instead, this project implements **constrained decoding**:
at every single generation step, the raw logits for all ~151,936 possible next tokens are
read directly from the model, only the tokens that keep the output a valid, schema-compliant
JSON value are kept as candidates, and the model picks only among those. The result is JSON
that is **always** syntactically valid and **always** matches the declared schema, while the
actual *content* (which function, which values) is still chosen by the model itself — never
by hardcoded keyword rules.

```
                     ┌─────────────────────────┐
                     │   natural language       │
                     │   "What is the product   │
                     │    of 3 and 5?"           │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Small_LLM_Model         │
                     │   (Qwen3-0.6B)             │
                     │   + constrained decoding   │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  {"prompt": "...",        │
                     │   "name": "fn_..." ,       │
                     │   "parameters": {...}}     │
                     └─────────────────────────┘
```

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk/` package copied next to this project (provides `Small_LLM_Model`)

### Project layout

```
project_root/
├── src/
│   ├── __init__.py
│   ├── __main__.py        ← CLI entry point (argparse + dependency check)
│   └── core.py             ← all constrained-decoding logic
├── llm_sdk/                ← provided model wrapper (copied, not modified)
├── data/
│   ├── input/               ← functions_definition.json, function_calling_tests.json
│   └── output/               ← function_calling_results.json (generated, not committed)
├── Makefile
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

### Installation

```bash
make install
# equivalent to: uv sync
```

### Running the program

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three arguments are optional — if omitted, the program defaults to the paths shown
above (`data/input/...` for reading, `data/output/...` for writing).

```bash
make run     # shortcut for the command above with default paths
make debug   # runs the program under pdb
make clean   # removes __pycache__ / .mypy_cache
make lint    # flake8 + mypy
make lint-strict
```

### Peer Review / Evaluation

After running the main program and generating `function_calling_results.json`, the evaluator
runs the Moulinette (grading script) to score the output against the private reference
dataset:

```bash
uv run python -m moulinette grade_student_answers \
    --set private \
    --student_answer_path ../data/output/function_calling_results.json
```

This step compares the generated `function_calling_results.json` file against the private
test set and produces the final grade. Run this **after** `make run` has produced a fresh
`data/output/function_calling_results.json` — the Moulinette does not regenerate results
itself, it only grades whatever file is already on disk at that path.

> **Note:** the exact Moulinette invocation (flags, working directory, dataset name) is
> defined by the official review guidelines for this activity. Confirm the command above
> still matches those guidelines before relying on it during a live evaluation.

## Resources

- [Anthropic — Prompt Engineering Overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Hugging Face — Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face — Guiding Text Generation with Constrained Decoding](https://huggingface.co/docs)
- [Python `re` module documentation](https://docs.python.org/3/library/re.html) — used for the
  numeric-literal validation regex
- [Pydantic documentation](https://docs.pydantic.dev/) — used for output validation
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259) — used as the
  reference for string-escaping rules in `generate_string`

### How AI was used

An AI assistant (Claude) was used throughout this project strictly as a **reviewer and
explainer**, not as a code author:

- **Diagnosing bugs from actual runtime output.** At every stage, real terminal output
  (tracebacks, printed results) was pasted in, and the assistant was asked to trace the bug
  back to the exact line responsible — for example, tracing a silent `NameError` on
  `number_tokens` back to a missing function call, or tracing a name-selection bias toward
  the longest function name back to un-normalized summed log-probabilities.
- **Explaining constrained-decoding concepts** (log-softmax, greedy vs. teacher-forced
  scoring, why averaging by path length removes length bias) before any code was written, so
  the implementation could be written from that understanding rather than copied.
- **Reviewing hand-written code changes line-by-line** after each edit, confirming or
  rejecting the reasoning before moving on, rather than the assistant proposing fixes to
  paste in directly.

All algorithm design decisions, the actual code, and every fix were written and typed by
the author. No code block was requested from or supplied by the AI for direct submission;
the assistant's role was limited to explaining *why* something was wrong and what the
underlying mechanism was, leaving the fix itself to be written and tested independently.

## Algorithm explanation

The pipeline builds the output JSON **one token at a time**, alternating between two modes:

```
              JSON so far: {"name":"
                       │
         ┌─────────────┴──────────────┐
         │                            │
   FIXED SYNTAX                 MODEL DECISION
   (no choice — the             (constrained: only
   next token(s) are             valid tokens are
   forced verbatim)               offered to the model)
         │                            │
   e.g. `{`, `"name"`,          e.g. which function
   `:`, `,`, `"parameters"`      name, which digit,
                                 which character
```

### Step 1 — Forced literals

Structural JSON syntax (`{`, `"name"`, `:`, `,`, `"parameters"`, closing `}`) is never
generated by the model — it is appended directly to the token sequence via `add_tokens`.
There is nothing to decide here, so this is the trivial case of constrained decoding: the
"allowed set" at each of these positions has exactly one member.

### Step 2 — Selecting the function name (teacher-forced scoring)

```
   candidates: fn_multiply_numbers | fn_is_even | fn_read_file | ...
                        │
        for EACH candidate independently:
                        │
        force-feed its real tokens one by one,
        record log P(token | context) at each step
                        │
        average the log-probabilities over the
        candidate's token length
                        │
        pick the candidate with the highest average
```

This is implemented in `score_function_candidates`. Rather than letting the model greedily
pick one token at a time and commit to whichever branch looked best *locally* (which can
lock onto a wrong path early and never recover), every known function name is scored as a
**complete, independent candidate** using teacher forcing, and the globally best-scoring name
is chosen. Log-probabilities (via `log_softmax`) are used instead of raw logits so scores are
on a comparable scale, and the score is the **average per token**, not the sum — this removes
a systematic bias toward longer function names (see *Challenges faced*).

### Step 3 — Generating parameter values

For each parameter, its type (from `functions_definition.json`) decides how it is generated:

```
   parameter type
        │
   ┌────┼─────┬──────────┐
   │    │     │          │
integer number boolean  string
   │    │     │          │
   │    │     │     free tokens, char-by-char
   │    │     │     JSON-escape aware, stops
   │    │     │     on an unescaped quote
   │    │  scored the same way as function
   │    │  names ("true" vs "false" as two
   │    │  candidates)
   │  digits/'.'/'-' only, regex-validated
   │  prefix at every step
 same as number, cast to int afterward
```

- **`integer` / `number`** (`generate_number`): at each step, only vocabulary tokens that keep
  the accumulated text a valid number prefix (`^-?\d*\.?\d*$`) are offered, plus the correct
  stop delimiter (`,` or `}`, depending on whether this is the last parameter). This guarantees
  the digits generated always form a syntactically valid JSON number.
- **`boolean`**: the two literals `"true"` and `"false"` are scored exactly like function
  names — via teacher-forced average log-probability — and the higher-scoring one is used.
- **`string`** (`generate_string`): the model is allowed to freely pick its top token at each
  step (rather than pre-banning characters), and the resulting text is tracked through a small
  state machine that understands JSON escaping (`\"`, `\\`, etc.) so it can tell an *escaped*
  quote (part of the content) from an *unescaped* quote (the real end of the string). The
  accumulated raw text is finally passed through `json.loads` to properly unescape it.

## Design decisions

- **Teacher-forced scoring over greedy trie-walking.** An earlier prototype walked a
  token-prefix trie greedily, committing to the locally best token at each branch. This
  reliably chose the wrong function whenever two names shared a prefix and diverged, because a
  locally-favored token could lock onto the wrong branch with no way to backtrack. Scoring
  every candidate as a whole, independent sequence and comparing the results removes this
  failure mode entirely.
- **Averaging log-probabilities, not summing them.** Summing (even after converting logits to
  log-probabilities) is length-biased: a longer name accumulates a larger total almost
  regardless of per-step confidence. Dividing by the number of tokens makes scores comparable
  across names of very different lengths.
- **A shared system-prompt context, built dynamically.** The model is given a system message
  listing every available function, its parameters, and its description — generated
  programmatically from `functions_definition.json`, never hardcoded. Without this, the model
  has no information to select the correct function from and effectively guesses.
- **ChatML formatting with an explicit, empty `<think></think>` block.** Qwen3 is a
  "thinking"-mode model; prompting it to emit `{` immediately, without first closing a
  thinking block, puts it outside its trained distribution and degrades output quality.
- **"Detect and cut" instead of "pre-ban" for string generation.** An early version pre-banned
  every token that could ever contain `"` or `\`, which also blocked legitimate regex syntax
  like `\d`. The final version lets the model choose freely and only intervenes once it sees
  what was actually chosen.
- **A calibrated no-match threshold (`NO_MATCH_LOG_PROB_THRESHOLD`).** If the best-scoring
  function candidate still scores below this threshold, the pipeline reports `name: None` with
  a `reason` instead of forcing a low-confidence guess into the output. The value (`-0.05`) was
  set by observing actual scores during debugging: valid, in-domain prompts scored around
  `-0.00001`, while an out-of-domain prompt scored around `-0.11` — `-0.05` sits between the
  two. This was checked on a small number of observed cases, not a systematic sweep across many
  in- and out-of-domain prompts, so it should be treated as a reasonable starting point rather
  than a rigorously validated boundary — see *Performance analysis*.

## Performance analysis

- **JSON validity:** 100% — every output, including the one semantically imperfect result, is
  syntactically valid JSON matching the declared schema, because the syntax itself is never
  left to free generation.
- **Speed:** each prompt requires one full forward pass per candidate function name (teacher
  forcing) plus one pass per generated token for parameters. With 6 candidate functions and
  short names/values, a full run over the test set completes well within the 5-minute budget
  stated in the requirements.
- **Reliability / error handling:** malformed or missing input files raise clear,
  human-readable errors (`_load_json_file`) instead of raw tracebacks; a missing dependency is
  caught before any model code runs (`src/__main__.py`); an empty or missing `"prompt"` field
  in a test entry is skipped with a logged reason instead of crashing the whole run; any
  unexpected exception while processing a single prompt is caught, logged, and recorded as a
  `None`-result entry so one bad prompt never stops the remaining ones.

### Known limitation

One test (a string containing nested quotes, e.g. `Say "hello" to {name}`) produces a trailing
extra quote character. This was investigated directly: the model's own confidence
(log-probability) in the closing quote was measured for both a correctly-terminating string
and this failing one, and the *genuinely correct* closing quote scored **lower** confidence
(-0.611) than the *spurious* one in the failing case (-0.168). Since the raw confidence signal
does not separate the two cases, no confidence-threshold hack was applied — doing so would
have "fixed" this one case while breaking others. This is documented here as a deliberate,
investigated decision rather than an unnoticed defect.

## Challenges faced

1. **The model was "blind" to the available functions.** Early prompts contained only the
   raw user question, with no information about which functions existed or what they did.
   Fixed by building a dynamic system-prompt listing every function, its parameters, and its
   description straight from `functions_definition.json`.

2. **Greedy trie-walking chose the wrong function in a systematic pattern.** Out of an early
   11-prompt test, 3 failures all collapsed onto the same wrong function whenever the correct
   answer required distinguishing between two functions with overlapping name prefixes.
   Traced to a token-by-token greedy walk that could never revisit an early wrong branch.
   Fixed by scoring every candidate name as a whole via teacher forcing (see *Algorithm
   explanation*, Step 2).

3. **The scoring fix initially collapsed onto the longest function name for every prompt.**
   Summing raw (or even log-) probabilities is length-biased. Fixed by switching to the
   **average** log-probability per token instead of the sum.

4. **Regex-generating parameters produced garbage or infinite repetition.** Pre-banning every
   token containing `"` or `\` also blocked legitimate regex syntax such as `\d` or `\s{2}`,
   forcing the model into degenerate workarounds. Fixed by switching from "pre-ban" to
   "detect what was chosen, then cut cleanly" — and by making sure the *only* independent stop
   signal is a literal quote character, since a comma or a closing brace can both legitimately
   appear *inside* real content (a sentence with a comma, or a regex quantifier like `\d{2}`).

5. **A CLI wiring gap silently made the whole program ignore its input file.** A hardcoded
   file path meant `--input` was accepted but never actually used — the program always read
   the same fixed path regardless of what was passed. Every single test failed with a
   prompt-mismatch symptom, which traced back not to the model or algorithm at all, but to
   `argparse` never being wired up. Fixed by adding proper `argparse` handling with the exact
   default paths specified in the project requirements.

6. **`"integer"` parameters were silently dropped.** The type-dispatch only handled
   `"number"` and `"string"`; any parameter typed `"integer"` in the schema matched neither
   branch and was silently skipped, producing an incomplete `parameters` object with no error
   at all. Fixed by adding an explicit `"integer"` branch and a final `else: raise ValueError`
   so any truly unsupported type fails loudly instead of vanishing silently.

7. **Backslash-heavy string values (Windows paths) came out double-escaped, and quoted
   content cut generation short.** A naive stop-on-quote check didn't distinguish an escaped
   quote (`\"`, part of the content) from a real terminating quote. Rewritten as a small
   character-level state machine that tracks an `escaped` flag, and the final raw text is
   passed through `json.loads` to unescape it correctly in one place instead of by hand.

8. **A stray leading space appeared at the start of some generated strings.** A known BPE
   tokenizer artifact: the model has learned that a space commonly follows a colon in natural
   text, and applied that habit even at the very start of a JSON string value. Fixed by
   stripping a leading space **only** from the first generated token, leaving genuine
   mid-string spaces (e.g. `SELECT * FROM users`) untouched.

9. **`UnboundLocalError` silently turned every string-typed result into `None`.** After fix
   #8 introduced an `is_first_token` flag, it was assigned *inside* an `if` block instead of
   before the loop. Python treats a variable as local to the whole function once it is
   assigned anywhere inside it, so the first read of `is_first_token` (in the `if` condition
   itself) raised `UnboundLocalError` before the assignment ever ran — and this exception was
   silently swallowed by the outer `try/except`, surfacing only as `name: None` with no
   visible traceback. Fixed by moving the initial assignment before the loop.

## Testing strategy

Testing was done by direct observation of real program output at every step, not by writing
a separate unit-test suite (the project only requires that a functional pipeline run
successfully against `data/input/`):

- An 11-prompt hand-built test set (`data/input/function_calling_tests.json`) covering all
  six declared functions, mixed parameter types (`number`, `integer`, `boolean`, `string`),
  and deliberately awkward cases: Windows file paths with backslashes, SQL strings containing
  commas and quotes, and a template string with nested quotes.
- After every code change, the full set was re-run and every single result line was read and
  compared by hand against the expected value — not just "did it crash", but "is this the
  semantically correct function, with the semantically correct value, for this exact wording".
- Regressions were caught this way multiple times (e.g. fix #8 above visually looked correct
  for non-string parameters while silently breaking every string parameter — this was only
  caught by re-running the *entire* set after each change, not just the case being fixed).
- The one remaining known failure (nested-quote trailing character, see *Performance
  analysis*) was investigated quantitatively — comparing the model's own log-probability
  confidence between a correct and an incorrect case — before deciding not to patch it with an
  unreliable heuristic.

## Example usage

```bash
$ uv run python -m src
result: {'prompt': 'What is the product of 3 and 5?', 'name': 'fn_multiply_numbers', 'parameters': {'a': 3.0, 'b': 5.0}}
result: {'prompt': 'Is 4 an even number?', 'name': 'fn_is_even', 'parameters': {'n': 4}}
result: {'prompt': 'Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years', 'name': 'fn_calculate_compound_interest', 'parameters': {'principal': 1234567.89, 'rate': 0.0375, 'years': 23}}
result: {'prompt': "Execute SQL query 'SELECT * FROM users' on the production database", 'name': 'fn_execute_sql_query', 'parameters': {'query': 'SELECT * FROM users', 'database': 'production'}}
result: {'prompt': 'Read C:\\Users\\john\\config.ini with latin-1 encoding', 'name': 'fn_read_file', 'parameters': {'path': 'C:\\Users\\john\\config.ini', 'encoding': 'latin-1'}}

Processed 11 / 11 prompts
Results written to: data/output/function_calling_results.json
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```
