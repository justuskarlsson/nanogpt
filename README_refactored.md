# Modded NanoGPT - Refactored

This repository contains a refactored version of the modded-nanogpt codebase, split into modular components for easier development and different use cases.

## File Structure

```
├── model.py          # Shared model architecture, optimizers, and utilities
├── pretraining.py    # Pretraining script (replaces train_gpt.py)
├── eval.py           # Evaluation and text generation script
├── chat.py           # Interactive chat interface
├── finetune.py       # Fine-tuning script for custom datasets
├── train_gpt.py      # Original monolithic training script (preserved)
└── README_refactored.md
```

## Components

### `model.py` - Shared Components
Contains all the reusable components:
- **Model Architecture**: `GPT`, `Block`, `CausalSelfAttention`, `MLP`, etc.
- **Optimizer**: Custom `Muon` optimizer and `zeropower_via_newtonschulz5`
- **Data Loading**: `distributed_data_generator`, `_load_data_shard`
- **Utilities**: `Hyperparameters`, `get_window_size_blocks`, `get_lr`, etc.
- **Helper Functions**: `create_model()`, `setup_optimizers()`, `load_checkpoint()`

### `pretraining.py` - Pretraining
For large-scale pretraining on raw text data:
```bash
# Single GPU
python pretraining.py

# Multi-GPU distributed training  
torchrun --nproc_per_node=8 pretraining.py
```

### `eval.py` - Evaluation & Generation
For evaluating trained models and generating text:
```bash
python eval.py
```
- Loads a checkpoint automatically
- Interactive text generation
- Customizable generation parameters

### `chat.py` - Interactive Chat
For chatting with your trained model:
```bash
python chat.py
```
Features:
- Interactive conversation interface
- Commands: `quit`, `clear`, `new`
- Maintains conversation context
- Temperature-based sampling

### `finetune.py` - Fine-tuning
For fine-tuning on custom conversational datasets:
```bash
python finetune.py
```
- Expects JSON format with conversation data
- Smaller learning rates and simplified optimizer
- Saves fine-tuned model checkpoint

## Usage Examples

### 1. Start with Pretraining
```bash
# Set up distributed environment
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

python pretraining.py
```

### 2. Evaluate Your Model
```bash
python eval.py
```
Update the `checkpoint_path` in `eval.py` to point to your trained model.

### 3. Chat with Your Model
```bash
python chat.py
```

### 4. Fine-tune for Specific Tasks
Create a `finetune_data.json` file:
```json
{
  "conversations": [
    {
      "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"}
      ]
    }
  ]
}
```
Then run:
```bash
python finetune.py
```

## Configuration

### Model Configuration
Edit the `Hyperparameters` class in `model.py`:
```python
@dataclass
class Hyperparameters:
    vocab_size = 50257
    train_seq_len = 48 * 1024
    val_seq_len = 4 * 64 * 1024
    num_iterations = 4000
    # ... other parameters
```

### Checkpoint Paths
Update checkpoint paths in each script:
```python
checkpoint_path = "logs/your-run-id/state_step004000.pt"
```

## Benefits of Refactoring

1. **Modularity**: Shared components in `model.py` avoid code duplication
2. **Specialization**: Each script focuses on a specific task
3. **Maintainability**: Changes to model architecture only need to be made in one place
4. **Extensibility**: Easy to add new scripts for different use cases
5. **Testing**: Individual components can be tested in isolation

## Dependencies

The same dependencies as the original project:
- torch
- transformers  
- python-dotenv
- (see requirements.txt)

## Migration Notes

- The original `train_gpt.py` is preserved for backward compatibility
- All functionality from the original script is maintained
- New scripts use the same model architecture and training procedures
- Checkpoint format remains compatible

## Adding New Scripts

To create a new script (e.g., for benchmarking):

1. Import from `model.py`:
```python
from model import create_model, load_checkpoint, Hyperparameters
```

2. Use the helper functions:
```python
model = create_model()
load_checkpoint(model, "path/to/checkpoint.pt")
```

3. Follow the same patterns as existing scripts

This modular structure makes it much easier to experiment with different training procedures, evaluation methods, and deployment scenarios while maintaining a clean, reusable codebase. 