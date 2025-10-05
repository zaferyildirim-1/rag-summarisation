from langchain.prompts import PromptTemplate

# Chunk-level / stuff-chain prompt (tuned for scientific text)
CHUNK_SUMMARY_PROMPT = PromptTemplate(
    template="""
Write a summary of the following text in about 750 tokens.
- Focus on the main contributions, methods, datasets, and results.
- Do not introduce information not present in the text.
- Keep the language objective and concise.
- Preserve key technical terms and abbreviations.
- If the text does not contain the requested information, say "No information found."

Text:
\"\"\"{text}\"\"\"

CONCISE SUMMARY:
""".strip(),
    input_variables=["text"],
)

# Combine / map-reduce prompt: combine per-chunk summaries into a final coherent output
COMBINE_SUMMARY_PROMPT = PromptTemplate(
    template="""
You are an expert researcher. Combine the following short summaries into one coherent final summary.
- Preserve facts and citations (add simple markers like [page X] where possible).
- Remove duplicate points and resolve contradictions by favoring more specific claims.
- Output: First a short title (1 line), then a 2-3 paragraph narrative summary, then 4-8 bullet key-takeaways.

Summaries:
{summaries}
""".strip(),
    input_variables=["summaries"],
)

# Refine prompt (for refine chain)
REFINE_PROMPT = PromptTemplate(
    template="""
We have an existing draft summary and a piece of new content. Update the draft to incorporate the new content:
Draft summary:
{existing_summary}

New content:
{new_content}

Return an improved summary; if new content contradicts the draft, update accordingly. If new content adds nothing, keep the draft unchanged.
""".strip(),
    input_variables=["existing_summary", "new_content"],
)

# Standard length instructions you can use from the UI
INSTRUCTION_SHORT = "Provide a concise summary in 3-5 sentences."
INSTRUCTION_MEDIUM = "Provide a summary with key bullets and a short paragraph (approx 150-300 words)."
INSTRUCTION_LONG = "Provide a detailed summary with a title and bullet points (~600-1000 words)."
