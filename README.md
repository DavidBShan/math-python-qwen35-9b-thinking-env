# Math Python Freesolo Environment

This is a Freesolo-compatible wrapper around PrimeIntellect's `math-python`
environment package. It preserves the upstream task contract:

- MATH-style examples.
- System prompt: use Python for calculations.
- Final answer must be inside `\boxed{}`.
- Reward uses `math-verify`.

The Flash configs in `configs/` are explicitly `thinking = true`, and the
served evaluator sends `chat_template_kwargs.enable_thinking = true`.
