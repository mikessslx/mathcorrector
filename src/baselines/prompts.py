SYSTEM_PROMPT = (
    "You are an expert in mathematical text error correction. "
    "Given a mathematical text that may contain errors in terminology, symbols, "
    "formulas, or logical statements, output the corrected version. "
    "Only fix genuine errors; preserve correct content exactly as-is. "
    "If the input has no errors, output it unchanged."
)

ZERO_SHOT_TEMPLATE = (
    "Please correct any errors in the following mathematical text. "
    "Output ONLY the corrected text, nothing else.\n\n{source}"
)

FEW_SHOT_HEADER = (
    "You are an expert in mathematical text error correction. Below are examples of "
    "mathematical texts with errors and their corrected versions.\n\n"
)

FEW_SHOT_EXAMPLE_TEMPLATE = "Input: {source}\nOutput: {target}\n\n"

FEW_SHOT_QUERY_TEMPLATE = "Input: {source}\nOutput:"
