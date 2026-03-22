#!/usr/bin/env python3
"""
Visualize CODI latent token analysis from decoded_latent.txt

Creates an HTML visualization showing:
- Question text
- For each latent: attended tokens and decoded (predicted) tokens
- Model prediction vs ground truth
- Color coding for attention to input vs latent tokens
"""

import re
import json
import html
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LatentStep:
    index: int
    attended_tokens: List[str]
    decoded_tokens: List[str]

@dataclass
class Example:
    question_id: int
    question: str
    cot: str
    answer: float
    latent_steps: List[LatentStep] = field(default_factory=list)
    prediction: str = ""
    predicted_answer: Optional[float] = None
    correct: bool = False

def parse_decoded_latent(filepath: str) -> List[Example]:
    """Parse the decoded_latent.txt file into structured examples."""
    examples = []
    current_example = None

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by question blocks
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # New question
        if line.startswith('Question') and '...' in line:
            if current_example is not None:
                examples.append(current_example)

            # Parse question ID
            match = re.match(r'Question(\d+)\.\.\.', line)
            qid = int(match.group(1)) if match else len(examples)
            current_example = Example(question_id=qid, question="", cot="", answer=0.0)
            i += 1
            continue

        # Question text
        if current_example and not current_example.question and line and not line.startswith('CoT=') and not line.startswith('decoded'):
            current_example.question = line.rstrip('...')
            i += 1
            continue

        # CoT and answer
        if line.startswith('CoT='):
            match = re.match(r'CoT=(.+), Answer=(.+)', line)
            if match and current_example:
                current_example.cot = match.group(1)
                current_example.answer = float(match.group(2))
            i += 1
            continue

        # Latent attended tokens
        if line.startswith('decoded') and "latent's attended tokens" in line:
            match = re.match(r"decoded (\d+)th latent's attended tokens \(top5\): \[(.+)\]", line)
            if match and current_example:
                idx = int(match.group(1))
                tokens_str = match.group(2)
                # Parse token list
                tokens = re.findall(r"'([^']*)'", tokens_str)

                # Next line should be decoded tokens
                i += 1
                if i < len(lines):
                    next_line = lines[i].strip()
                    match2 = re.match(r"decoded \d+th latent \(top5\): \[(.+)\]", next_line)
                    if match2:
                        decoded_tokens = re.findall(r"'([^']*)'", match2.group(1))
                    else:
                        decoded_tokens = []
                        i -= 1  # back up
                else:
                    decoded_tokens = []

                current_example.latent_steps.append(LatentStep(
                    index=idx,
                    attended_tokens=tokens,
                    decoded_tokens=decoded_tokens
                ))
            i += 1
            continue

        # Before answer token attention
        if line.startswith("decoded before answer token's attended"):
            # Skip this line, we already have all latent steps
            i += 1
            continue

        # Model prediction
        if line.startswith('Model Prediction:'):
            if current_example:
                pred = line.replace('Model Prediction:', '').strip()
                current_example.prediction = pred
                # Extract number from prediction
                numbers = re.findall(r'-?\d+\.?\d*', pred)
                if numbers:
                    current_example.predicted_answer = float(numbers[-1])
                    current_example.correct = (current_example.predicted_answer == current_example.answer)
            i += 1
            continue

        i += 1

    # Don't forget last example
    if current_example is not None:
        examples.append(current_example)

    return examples

def token_to_html(token: str, is_latent: bool = False) -> str:
    """Convert a token to HTML with appropriate styling."""
    escaped = html.escape(token)

    if token.startswith('<LAT'):
        # Latent token reference - pink/red
        return f'<span class="latent-ref">{escaped}</span>'
    elif token == '<PAD>':
        return f'<span class="pad-token">{escaped}</span>'
    elif token == '<BOT>':
        return f'<span class="special-token">{escaped}</span>'
    elif token == '<EOT>':
        return f'<span class="special-token">{escaped}</span>'
    else:
        # Regular input token - yellow/gold
        return f'<span class="input-token">{escaped}</span>'

def example_to_html(ex: Example) -> str:
    """Convert an example to HTML."""
    status_class = "correct" if ex.correct else "incorrect"

    html_parts = [f'''
    <div class="example {status_class}">
        <div class="question-header">
            <span class="question-id">Question {ex.question_id}</span>
            <span class="status {'✓' if ex.correct else '✗'}">{('Correct' if ex.correct else 'Incorrect')}</span>
        </div>
        <div class="question-text">{html.escape(ex.question)}</div>
        <div class="cot-answer">
            <span class="cot">Golden CoT: {html.escape(ex.cot)}</span>
            <span class="answer">Answer: {ex.answer}</span>
        </div>
        <div class="latent-steps">
    ''']

    for step in ex.latent_steps:
        attended_html = ', '.join(token_to_html(t) for t in step.attended_tokens)
        decoded_html = ', '.join(f'<span class="decoded-token">{html.escape(t)}</span>' for t in step.decoded_tokens)

        # Highlight the top decoded token
        top_decoded = step.decoded_tokens[0] if step.decoded_tokens else ""

        html_parts.append(f'''
            <div class="latent-step">
                <div class="latent-label">z<sub>{step.index}</sub></div>
                <div class="latent-content">
                    <div class="attended">
                        <span class="label">Attended:</span> [{attended_html}]
                    </div>
                    <div class="decoded">
                        <span class="label">Decoded:</span> [{decoded_html}]
                    </div>
                </div>
                <div class="top-prediction">{html.escape(top_decoded)}</div>
            </div>
        ''')

    html_parts.append(f'''
        </div>
        <div class="model-prediction">
            <span class="label">Model Prediction:</span> {html.escape(ex.prediction)}
        </div>
    </div>
    ''')

    return ''.join(html_parts)

def generate_html(examples: List[Example], output_path: str):
    """Generate the full HTML page."""

    correct_count = sum(1 for ex in examples if ex.correct)
    total_count = len(examples)
    accuracy = 100 * correct_count / total_count if total_count > 0 else 0

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CODI Latent Token Visualization</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats .accuracy {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .filters {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .filters button {{
            padding: 8px 16px;
            margin: 0 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        .filters button.active {{
            background: #1976d2;
            color: white;
        }}
        .filters button:not(.active) {{
            background: #e0e0e0;
        }}
        .example {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .example.correct {{
            border-left: 4px solid #4caf50;
        }}
        .example.incorrect {{
            border-left: 4px solid #f44336;
        }}
        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .question-id {{
            font-weight: bold;
            font-size: 18px;
            color: #1976d2;
        }}
        .status {{
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 14px;
        }}
        .correct .status {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .incorrect .status {{
            background: #ffebee;
            color: #c62828;
        }}
        .question-text {{
            background: #fff8e1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            line-height: 1.5;
        }}
        .cot-answer {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }}
        .cot {{
            color: #666;
            font-family: monospace;
        }}
        .answer {{
            font-weight: bold;
            color: #1976d2;
        }}
        .latent-steps {{
            margin-bottom: 15px;
        }}
        .latent-step {{
            display: grid;
            grid-template-columns: 50px 1fr 80px;
            gap: 10px;
            align-items: center;
            padding: 10px;
            margin-bottom: 8px;
            background: #fafafa;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        .latent-label {{
            font-weight: bold;
            font-size: 16px;
            color: #7b1fa2;
            text-align: center;
        }}
        .latent-content {{
            font-family: monospace;
            font-size: 13px;
        }}
        .attended, .decoded {{
            margin-bottom: 5px;
        }}
        .label {{
            font-weight: bold;
            color: #666;
        }}
        .input-token {{
            background: #fff59d;
            padding: 2px 4px;
            border-radius: 2px;
            margin: 0 2px;
        }}
        .latent-ref {{
            background: #f8bbd9;
            padding: 2px 4px;
            border-radius: 2px;
            margin: 0 2px;
        }}
        .pad-token {{
            background: #e0e0e0;
            padding: 2px 4px;
            border-radius: 2px;
            margin: 0 2px;
            color: #666;
        }}
        .special-token {{
            background: #b3e5fc;
            padding: 2px 4px;
            border-radius: 2px;
            margin: 0 2px;
        }}
        .decoded-token {{
            background: #c8e6c9;
            padding: 2px 4px;
            border-radius: 2px;
            margin: 0 2px;
        }}
        .top-prediction {{
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #2e7d32;
            background: #e8f5e9;
            padding: 5px;
            border-radius: 4px;
        }}
        .model-prediction {{
            padding: 10px;
            background: #e3f2fd;
            border-radius: 4px;
            font-family: monospace;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <h1>CODI Latent Token Visualization</h1>

    <div class="stats">
        <div class="accuracy">{accuracy:.1f}% Accuracy</div>
        <div>{correct_count} / {total_count} examples correct</div>
    </div>

    <div class="filters">
        <button class="active" onclick="filterExamples('all')">All ({total_count})</button>
        <button onclick="filterExamples('correct')">Correct ({correct_count})</button>
        <button onclick="filterExamples('incorrect')">Incorrect ({total_count - correct_count})</button>
    </div>

    <div id="examples">
'''

    for ex in examples:
        html_content += example_to_html(ex)

    html_content += '''
    </div>

    <script>
        function filterExamples(filter) {
            const examples = document.querySelectorAll('.example');
            const buttons = document.querySelectorAll('.filters button');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            examples.forEach(ex => {
                if (filter === 'all') {
                    ex.classList.remove('hidden');
                } else if (filter === 'correct') {
                    ex.classList.toggle('hidden', !ex.classList.contains('correct'));
                } else if (filter === 'incorrect') {
                    ex.classList.toggle('hidden', !ex.classList.contains('incorrect'));
                }
            });
        }
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Generated visualization: {output_path}")
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize CODI latent tokens')
    parser.add_argument('--input', '-i', default='outputs/decoded_latent.txt',
                        help='Input decoded_latent.txt file')
    parser.add_argument('--output', '-o', default='outputs/latent_visualization.html',
                        help='Output HTML file')
    args = parser.parse_args()

    print(f"Parsing {args.input}...")
    examples = parse_decoded_latent(args.input)
    print(f"Found {len(examples)} examples")

    generate_html(examples, args.output)

if __name__ == '__main__':
    main()
