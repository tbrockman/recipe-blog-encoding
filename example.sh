python main.py encode \
  --message "https://www.nokings.org/" \
  --stego-file stego_output.txt \
  --model "models/Qwen3-32B-Q4_K_M.gguf" \
  --prompt "<|im_start|>user
You are a blog author writing your stereotypical recipe introduction, aiming to maximize the chances of your recipe being seen by gaming your articles SEO, without being too verbose. Output only the recipe introduction and nothing else, using a maximum of 12 paragraphs. Choose a random recipe.<|im_end|>
<|im_start|>assistant
<think>

</think>"