# Human rating rubric: condition briefs (Component E)

**Who rates:** at least two raters (Luke plus one team member), working independently. Report inter-rater agreement (weighted Cohen's κ, or % within one point).

**What:** each row of `results/rating_<run>.csv`. Rows are shuffled so the engines (template, qwen3, llama3.1) are not grouped together.

**How:**
1. Open the matching brief in `results/briefs_<run>.jsonl`, using the `markdown` field.
2. Rate the brief on the three criteria below.
3. For the accuracy score, check the brief against its `payload`.
4. Do not look at `payload.hindsight` until the brief has been rated. The brief is a prediction, and judging it against what actually happened measures the ML model, not the writing.

| Score | Accuracy | Usefulness | Clarity |
|---|---|---|---|
| **5** | Every statement is supported by the payload, and uncertainty is stated correctly | An asset manager could act on it directly | Reads naturally and is concise, with no jargon beyond PCI, IRI and rut depth |
| **4** | One minor imprecision that doesn't change the meaning | Useful, with one small gap | Clear, but slightly wordy or awkward |
| **3** | One statement is misleading, or a caveat is missing or understated | Useful only with the payload beside it | Understandable, but it takes effort |
| **2** | Several misleading statements, or a driver described as a cause | Of little practical use | Hard to follow, or reads like a list of fields |
| **1** | Contradicts the payload | Would mislead a decision | Unreadable |

**Record in `comment`:**
- any statement you judged wrong, with the reason
- anything you would want added

**Report in the thesis:**
- the mean and median for each criterion by engine
- inter-rater agreement
- the automated metrics from `summary_<run>.csv`: first-try pass rate, invented numbers, fallback rate and latency
