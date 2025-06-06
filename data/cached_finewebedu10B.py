import os
import sys
from huggingface_hub import hf_hub_download


# Download the GPT-2 tokens of FinewebEDU10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname):
    local_dir = os.path.join(os.path.dirname(__file__), "finewebedu10B")
    if not os.path.exists(os.path.join(local_dir, fname)):
        hf_hub_download(
            repo_id="kjj0/finewebedu10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
        )


get("finewebedu_val_%06d.bin" % 0)
num_chunks = 99  # full FinewebEDU10B. Each chunk is 100M tokens
if len(sys.argv) >= 2:  # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks + 1):
    get("finewebedu_train_%06d.bin" % i)

# TODO: Use this for our pretraining.
# Enough tokens for chinchilla 1B model constraint.
# But wait, that's 20B tokens.. So maybe downsize to 500M model?
# For 130M model, 48k tokens pers step -> 48M tokens / 1k
# To reach 2B, we need 40k steps. Like 4 hours or something.
# So:
# 1. download 2B finewebedu
# 2. downscale to original 130M model.
# 3. train for 40k steps.
