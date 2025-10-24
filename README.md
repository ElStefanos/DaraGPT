# Due to lack of System resources this model is temporary halted in development. I've hit the point where optimisation doesn't help anymore. These are the limitations of OpenCL and my hardware that unfortunately I have to use.

# DaraGPT

**DaraGPT** is an experimental GPT-style language model written entirely in **C#** with **OpenCL acceleration**.  
It is designed for local, privacy-preserving training and experimentation on personal hardware, without relying on cloud resources.  
The system supports tokenization, model training, fine-tuning, and inference through a modular and transparent architecture.

---

## Key Features

- Full **from-scratch training** on local text datasets/
- **GPU acceleration** using OpenCL (Intel, AMD, and NVIDIA compatible)
- **BPE tokenizer** with binary storage format (`.tokbin`)
- Modular **Transformer architecture** with configurable depth and heads
- **Parallel training** per file, each producing an independent sub-model `TODO`
- **Automatic model merging** (averaged weights) `TODO`
- Optional **fine-tuning** phase on combined datasets `TODO`
- **Checkpointing system** for training resumption `partially implemented`
- **CPU/GPU fallback** (runs even without a GPU (NOT RECOMMENDED)) 
- Supports **dynamic context size** (typically 256–1024 tokens)  `TODO`

---

## Project Structure

DaraGPT/ <br>
├── Data/ -> This is where tha data is being pulled from <br>
├── GPUKernels/ # OpenCL kernel source files <br>
├── Model/<ModelName_x>/ -> Stored <name>.(.tokbin,.seqbin,.modbin, .epmodbin (optional) <br>
├── FinalModels/ -> Merged and fine-tuned models <br>

## Configuration

Example `Config/config.json` file:

```json
{
  "ModelName": "DaraGPT",
  "ModelSavePath": "./Model/",
  "DataPath": "./Data/",
  "DModel": 192,
  "Head": 8,
  "Layers": 6,
  "ContextSize": 512,
  "VocabSize": 5000,
  "LearningRate": 0.0025,
  "DevicePreference": "INTEL"
}
```
## Training Workflow
### 1. Dataset Preparation

Place text files into the Data/ directory.
Subdirectories are supported (e.g., /Books, /Wiki, /Subtitles, /News).

### 2. Tokenization
The tokenizer produces a .tokbin file used across all model training.
The tokenizer iz BPE based.

### 4. File-Based Training
Each dataset file is trained independently, creating separate models. (Implemented but is meant to be used with parallel training)

### 5. Parallel Training - TODO

Run multiple trainers in parallel to fully utilize GPU resources.

### 6. Model Merging - TODO

After all sub-models are trained, merge them into a single unified model.

### 7. Fine-tuning (optional) - TODO
Fine-tune the merged model on a representative corpus.

## Chatting

Ability to run interactive inference with the trained model.
HTTP server will be implemented in the future.

## Performance

### *NOTE: This was sampled on avarage file size of a 125kb and avarage time on really slow systems.*

| GPU / CPU                  | Context | dModel | Layers | VRAM Usage | Training Speed           |
| -------------------------- | ------- | ------ | ------ | ---------- | ------------------------ |
| Intel Iris Xe (integrated) | 256     | 192    | 6      | ~0.8 GB    | ~2.5–4 min/MB (parallel) |
| AMD R9 380 (2gb)               | 512     | 256    | 8      | ~1 GB    | ~1.2 min/MB              |


## File formats

| Extension | Description                                |
| --------- | ------------------------------------------ |
| `.tokbin` | Binary tokenizer file (token ↔ ID mapping) |
| `.seqbin` | Optional sequence data for checkpoints     |
| `.modbin`    | Model weights                              |
| `.json`   | Configuration file                         |

## Feedback and Fine-Tuning Loop (Planned)

DaraGPT includes a planned reinforcement component where users can rate responses (positive or negative).
These ratings are stored in a local SQLite database.

# Roadmap


| Phase                       | Status      | Description                            |
| --------------------------- | ----------- | -------------------------------------- |
| Core architecture           | Complete    | Model, tokenizer, GPU backend          |
| Parallel training & merging | Partially complete    | Multi-file training and merge system   |
| Fine-tuning module          | Planned | Manual and automated fine-tune support |
| Feedback-based learning     | Planned     | RLHF-style reward optimization         |
| Monitoring GUI              | Planned     | Real-time epoch/loss visualization     |
| Model packaging             | Planned     | Export and deployment tools            |

# Notes

**The project is for educational and research purposes only.

Performance varies depending on GPU driver and OpenCL implementation.

Tokenizer and model formats are custom and not compatible with OpenAI or Hugging Face models.

Training and inference are fully offline.**

# License

This project is open for research and personal use.
Author: Stefan Crkvenjakov


