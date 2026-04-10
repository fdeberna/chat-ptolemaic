import json
from openai import OpenAI

client = OpenAI()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "quality_score": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": "0=incoherent or degenerate, 1=partly coherent but flawed, 2=coherent and meaningful"
        },
        "stance_label": {
            "type": "string",
            "enum": [
                "geocentric",
                "heliocentric_or_earth_moves",
                "ambiguous",
                "no_relevant_claim"
            ]
        },
        "heliocentric_label": {
            "type": "integer",
            "enum": [0, 1],
            "description": "1 only if the text clearly suggests Earth moves or meaningfully entertains a sun-centered account"
        },
        "reason": {
            "type": "string"
        }
    },
    "required": [
        "quality_score",
        "stance_label",
        "heliocentric_label",
        "reason"
    ],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are an expert evaluator of short medieval-style astronomy passages.

Judge the generated text according to this rubric:

- quality_score:
  0 = incoherent, degenerate, repetitive, or not meaningfully interpretable
  1 = partly coherent but confused, contradictory, or weak
  2 = coherent, readable, and meaningfully responsive

- stance_label:
  geocentric = clearly supports Earth at rest / geocentric order
  heliocentric_or_earth_moves = clearly suggests Earth moves or that a sun-centered explanation is being entertained
  ambiguous = astronomy-related but unclear
  no_relevant_claim = no meaningful astronomical claim

- heliocentric_label:
  1 only if the passage clearly suggests Earth moves or meaningfully entertains a heliocentric explanation.
  Do NOT label heliocentric just because the text contains generic motion language, confused phrasing, or nonsense.

Return only the schema fields.
"""

# prompt_text = "Whether the Earth is at rest at the center of the universe"
# generated_text = (
#     "It may be asked whether the Earth moves. I answer that it does not; "
#     "for heavy bodies seek the center and there come to rest."
# )

# prompt_text = "Whether the Sun moves around the Earth or the Earth around the Sun"
# generated_text = (
#     "Some might hold that the daily motion belongs not to the heavens but to the Earth itself, which by its turning causes the rising and setting of the stars."
# )

# prompt_text = "What is the cause of the annual motion of the Sun?"
# generated_text = (
#     "The annual motion may be understood in diverse ways, according as one considers the appearances or the hidden causes, and the matter is not easily resolved."
# )

prompt_text = "Concerning the order of the spheres, it is known that"
generated_text = (
    "The line of the center is the center of the line through the center and thus the center remains in the center of the center."
)


user_text = f"""Prompt:
{prompt_text}

Generated text:
{generated_text}
"""

response = client.responses.create(
    model="gpt-5.4-mini",
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "judge_result",
            "schema": JUDGE_SCHEMA,
            "strict": True,
        }
    },
)

print(response.output_text)
print()
print(json.loads(response.output_text))