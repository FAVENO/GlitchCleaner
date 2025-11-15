# GlitchCleaner

This repository contains the source code for the paper "GlitchCleaner: Lightweight Glitch Tokens Repairing by Lossless Gated LoRA in Large Language Models". In this work, we propose GlitchCleaner, a lightweight and
lossless method for repairing glitch tokens—inputs that trigger abnormal outputs in large language models. Experimental results across multiple mainstream models demonstrate that our method achieves an average repair rate of 86.88\%, representing an improvement of over 30\% compared to existing approaches, while ensuring lossless preservation of the model’s baseline capabilities and causing negligible impact on inference speed.

<!-- 📄 [View Paper]() -->

## Code Organization
This repository contains three directories: `Fine-tuning`, `Glitchtokens`, and `LoRA-parameter`, plus the tutorial notebook. The `Fine-tuning` directory includes the scripts used for training, the `Glitchtokens` directory stores the detected glitch tokens, and the `LoRA-parameter` directory provides the repaired LoRA weights.

## Quick Start
You can quickly reproduce part of the experimental results with `Tutorial.ipynb`. It loads the repaired weights from `LoRA-parameter` to evaluate both the repair success rate and the preserved baseline capabilities of the model.

If you want to fine-tune a model yourself, refer to the `Fine-tuning` directory.

## Appendix
See `Appendix.pdf` for more details.




