---
name: analyze-output
description: Read and analyze model output files - logs, predictions, metrics, decoded tokens, or any structured text output from ML experiments. Use when the user asks to look at results, check outputs, or analyze experiment data.
argument-hint: "<path to output file>"
---

# Analyze Model Output

Read and analyze the output file at `$ARGUMENTS`.

## Steps

1. **Read the file** and determine its format (log, CSV, JSON, structured text, etc.)
2. **Extract key metrics**: accuracy, loss, counts, or any numerical summaries
3. **Identify patterns**: recurring errors, trends across examples, distribution of results
4. **Provide a concise summary** with:
   - Overall statistics (accuracy, averages, distributions)
   - 2-3 representative examples showing interesting behavior
   - Any anomalies, failure modes, or notable patterns
5. **Suggest next steps** if relevant (parameter changes, further analysis, visualization)

Keep analysis concise and data-driven. Show numbers, not just descriptions.
