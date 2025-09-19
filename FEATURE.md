# Finetune

**Download data**
Pick a good dataset for finetuning instruction llm. Think there is one on huggingface?

**Organize data into finetuning ready**
- Mask for delineating between question and answer. No loss on question part.
- In a format that we can later read a batch from.
- Tokenize?


**Training**
Like model.py and pretraining.py, but modified for finetuning.

**Eval**
- Start with vibe_test.py (add finetuned model)
- Then do non-STEM benchmarks, like SWAGsomething..