"""Hello world: load a small model and ask it one math question.

Run:  python src/hello.py            (defaults to Qwen/Qwen3-0.6B)
      python src/hello.py <hf_repo>
"""
import sys

from mlx_lm import generate, load

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
model, tokenizer = load(repo)  # downloads to ~/.cache/huggingface on first run

question = ("Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?")
messages = [{"role": "user", "content": question + "\nThink step by step, then give the final answer after '####'."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=True)  # verbose prints the output + stats
