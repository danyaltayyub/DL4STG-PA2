"""Convert ai_usage_log.md (the working log) into sections/ai_usage.tex for the report appendix."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def latex(text):
    text = text.replace("\\", r"\textbackslash{}")
    for char in "&%$#_{}":
        text = text.replace(char, "\\" + char)
    text = re.sub(r"`([^`]*)`", r"\\texttt{\1}", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w\\])_([^_]+)_(?!\w)", r"\\textit{\1}", text)
    return text.replace("–", "--").replace("→", r"$\to$").replace("≈", r"$\approx$")


def main():
    rows = []
    for line in (HERE / "ai_usage_log.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            rows.append(cells)
    body = "\n".join(
        f"{n} & {latex(stage)} & {latex(prompt)} & {latex(output)} & {latex(edits)} \\\\ \\midrule"
        for n, stage, prompt, output, edits in rows)
    tex = r"""\section{Use of generative AI}

Tool: Claude (Anthropic) via Claude Code, used throughout as a coding, analysis and writing
assistant; it wrote and executed code in the student's environment and drafted report text.
In line with the course AI policy, each row gives the prompt (paraphrased in formal language,
preserving its substance), what the tool produced, and what the student changed or decided.
Every number in this report is produced by, and reproducible from, the code in the repository.

{\footnotesize
\begin{longtable}{p{0.025\textwidth}p{0.11\textwidth}p{0.26\textwidth}p{0.26\textwidth}p{0.21\textwidth}}
\toprule
\# & Stage & Prompt (paraphrased) & Output & Student edits and decisions \\ \midrule
\endhead
""" + body + "\n\\bottomrule\n\\end{longtable}\n}\n"
    (HERE / "sections" / "ai_usage.tex").write_text(tex)
    print(f"{len(rows)} rows written")


if __name__ == "__main__":
    main()
