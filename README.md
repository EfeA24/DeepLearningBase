Deep Learning Base — Reusable Code Warehouse
📌 Branch notice: The reusable source code is currently maintained on the master branch. Check out or select master to access the latest code in this repository.
git clone --branch master <YOUR-GITHUB-REPOSITORY-URL>
Replace <YOUR-GITHUB-REPOSITORY-URL> with this repository's GitHub clone URL. If you have already cloned the repository, switch to the branch with git switch master.

A personal, modular warehouse of reusable deep learning code for computer vision, recurrent networks, and NLP/Transformers. It is designed to make future project work faster by keeping common components ready to inspect, copy, adapt, and reuse.
This repository is intentionally organized as a collection of templates and copyable code snippets, rather than as one application or a single installable framework. When starting a new project, take only the files you need, copy them into that project, and adapt their interfaces to its data, task, and runtime.

Table of contents
- Why this repository exists
- How to use the code
- Repository areas
- Template catalogue
- Design principles
- Python import policy
- Purpose comments in Python files
- Dependencies
- A typical copy-and-adapt workflow
- Scratch implementations and reference code
- Scope and expectations
- Possible future additions
- License
     Why this repository exists
Deep learning projects often repeat the same engineering work: dataset wrappers, tokenization, padding, masks, model blocks, training steps, metrics, checkpoint handling, and inference helpers. This repository keeps reusable versions of those pieces in one place.
The goal is to:
- Reduce repeated boilerplate across projects.
- Keep model components small enough to copy and adapt independently.
- Preserve educational and scratch implementations for understanding how models work.
- Keep task-specific components separate across vision, sequence modelling, and NLP.
- Make each file understandable when viewed on GitHub or copied without its original context.
- Leave room to change a component for a particular dataset, architecture, or deployment target.
The repository is a working code warehouse. It is not intended to impose one universal architecture or training framework on every future project.

How to use the code
1. Make sure you are viewing the master branch.
2. Browse the template that matches your task.
3. Copy only the components needed for your project.
4. Read the file's opening Purpose comment and its API/docstring before using it.
5. Check the expected tensor shapes, label format, padding behavior, box coordinates, and dependencies.
6. Adapt paths, configuration, data schema, and model interfaces in the new project.
7. Validate the copied code in the context of that project's data and environment.
A copied component is a starting point. Its surrounding project may need additional files, dependencies, or integration code.

Repository areas
The warehouse is grouped by problem family, with common utilities collected separately:
Area	Intended contents
00-common/	Optional cross-project snippets for training, optimizers, schedulers, checkpointing, precision, distributed execution, metrics, logging, reproducibility, and utilities.
01-cv-classification-base/	Image classification datasets, transforms, CNN components, losses, metrics, training, checkpoints, and inference.
02-cv-detection-base/	Object detection datasets and annotation parsers, box operations, detector components, losses, evaluation, inference, and export helpers.
03-rnn-base/	Vanilla RNN reference implementations and reusable sequence data utilities.
04-lstm-gru-base/	LSTM and GRU reference implementations, sequence pooling, attention, state handling, and training helpers.
05-transformer-encoder-base/	Encoder-style Transformer components for classification, token classification, embeddings, and masked language modelling.
06-transformer-decoder-base/	Causal decoder components for language modelling and text generation.
07-transformer-seq2seq-base/	Encoder-decoder components for translation, summarization, and other text-to-text tasks.


The precise contents of a template can evolve as reusable components are added. Treat the directories and individual files on master as the current source of truth.


cv_classification_base
For image classification experiments and project starts. The area is organized around:
- Data: folder/CSV/custom datasets, loaders, image transforms, augmentation, samplers, splitting, and collators.
- Models: educational CNNs, backbone adapters, convolutional/residual blocks, pooling, initialization, and classification heads.
- Learning components: cross-entropy, focal loss, label smoothing, BCE, optimizer/scheduler examples, and classification metrics.
- Project utilities: train/validation steps, early stopping, inference, checkpoint loading/saving, device and reproducibility helpers.
Typical use: copy a dataset adapter, transforms, a backbone or CNN block, a classification head, and the training/evaluation helpers required by a new classification task.

cv_detection_base
For object detection data and model-building experiments. The area covers:
- Detection dataset contracts and adapters for YOLO-, COCO-, VOC-, or custom-style annotations.
- Annotation parsing, class mapping, and conversion between bounding-box formats.
- Box-aware transforms and optional augmentation scaffolds.
- Backbone, neck, head, anchor, and detector extension points.
- IoU-family geometry, NMS, detection losses, precision/recall, and mAP-related evaluation.
- Inference post-processing, visualization, ONNX export, and optional TensorRT integration helpers.
Typical use: copy the annotation parser and box conversions that match the project, then adapt the dataset target contract, detector, loss, and post-processing to the selected detection architecture.

rnn_base
For vanilla recurrent networks and introductory sequence modelling. The area includes sequence datasets, small tokenizer/vocabulary helpers, padding and windowing, RNN scratch/reference code, hidden-state utilities, sequence heads, masked losses, truncated BPTT, and autoregressive generation helpers.
Typical use: copy the sequence preparation functions and either a scratch RNN implementation for study or a PyTorch RNN adapter for a project baseline.
lstm_gru_base
For gated recurrent networks. This area separates LSTM and GRU cells and model wrappers, with optional attention, mask-aware pooling, hidden/cell state helpers, sequence training, and generation components.
Typical use: choose an LSTM or GRU implementation, copy only the sequence preparation/pooling/state pieces required, and define the project's own state-reset and batching policy.
transformer_encoder_base
For bidirectional encoder tasks such as text classification, NER/POS-style token classification, sentence-pair tasks, embeddings, and masked language modelling. The area is separated into datasets, tokenizer adapters, collators/masks, embeddings, attention, feed-forward layers, normalization, encoder blocks, heads, and task losses.
Typical use: copy an encoder block/stack and only the task-specific data and head components. Token-to-word label alignment for token classification must be selected and validated for the tokenizer and dataset in use.

transformer_decoder_base
For causal language modelling and autoregressive generation. The area covers text/instruction data preparation, chunking and packing, causal masks, token/position embeddings, rotary embeddings, decoder attention, grouped-query attention, KV-cache helpers, feed-forward variants, decoder blocks, sampling, and causal-LM training/inference.
Typical use: use the decoder components as a reference or starting point, then match tokenizer vocabulary, context length, attention implementation, cache behavior, and training objective to the selected model.

transformer_seq2seq_base
For source-to-target problems such as translation, summarization, and structured text generation. The area separates source and target processing, tokenizers, teacher-forcing shifts, masks, encoder/decoder blocks, cross-attention, decoding strategies, training, and inference facades.
Typical use: define a consistent source/target token contract, copy the corresponding collator and model pieces, then select greedy decoding, beam search, or sampling according to the task.


common
Optional cross-project snippets cover training state, gradient accumulation/clipping, optimizers, learning-rate schedules, checkpoints, distributed setup examples, mixed precision, metrics, logging, reproducibility, and small utilities.
These files are kept available for copying. The model templates should not require the repository to be installed as a package or import 00-common at runtime. Copy the common file you need into your project and adjust it as needed.
⚙️ Design principles
Copyable by design
Files should have a clear, narrow responsibility. Avoid relying on a large hidden framework, import-time registration, or assumptions that every neighboring module is present.
Explicit interfaces
Data formats and tensor contracts matter. Before reuse, check and preserve details such as:
- Tensor dimensions and whether batches are batch-first.
- Image channel order and value range.
- Detection box format (xyxy or xywh) and pixel or normalized coordinates.
- Token padding ids, sequence lengths, masks, and ignored labels.
- Whether model outputs are raw logits or probabilities.
- Whether metrics use macro, micro, or weighted averaging.
- When a scheduler is expected to step: per batch, optimizer update, or epoch.
No hidden work on import
Importing a module should not start training, download a model, read a dataset, scan directories, configure global logging, initialize a process group, load a checkpoint, or select a device as a side effect.
Keep optional integrations optional
Adapters for external libraries should be isolated. A project should not need computer-vision, tokenizer, tracking, or export packages merely to import an unrelated component.
Keep examples honest
A scratch implementation is educational reference code. An adapter is glue for a selected dependency. A scaffold marks the places a future project must implement. A filename alone does not imply that a component is production-ready or compatible with every library version.

Python import policy
Torch is the only dependency allowed to be imported at module scope. All other imports should be lazy.
Place optional third-party imports inside the function, method, or factory that uses them. This applies to packages such as NumPy, Pillow, torchvision, OpenCV, Transformers, tokenizers, datasets, scikit-learn, TensorBoard, ONNX, and TensorRT. Project-local sibling modules should also not be imported eagerly at module scope.
Example pattern:
# Purpose: Adapts a caller-supplied Hugging Face tokenizer without loading Transformers for unrelated imports.

class HuggingFaceTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name_or_path, **kwargs):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install transformers to use this adapter."
            ) from exc

        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))
A Torch model may import Torch at the top of its file:
# Purpose: Maps pooled image features to raw class logits.
import torch
from torch import nn
Package initializers should remain minimal and must not import every implementation module for convenience. Prefer direct imports from the component file when integrating copied code.
Purpose comments in Python files
Each .py file should begin with a concise English comment that explains why the file exists and what reusable component it provides. This is intended to act as a small README when a file is viewed or copied on its own.
# Purpose: Parses YOLO normalized bounding-box rows into validated annotation records.
# Purpose: Builds a padding mask for variable-length encoder input batches.
# Purpose: Provides an educational scratch LSTM cell for inspecting gate-level recurrence.
For an unfinished scaffold, say so directly:
# Purpose: Provides an adaptation scaffold for a project-specific detection dataset.
Avoid vague comments such as # Helper functions or # This file contains code.
Dependencies
There is no single dependency set required for every file. Dependencies depend on which component is copied and used.
- PyTorch is used by Torch model/training components.
- Optional integrations may require packages for image loading and transforms, pretrained model adapters, tokenization, metric computation, logging, distributed execution, or export.
- Install only the dependencies required by the selected files and the target project.
- Check the component's local imports/docstring and the target library's version documentation before use.
- No component should download weights, datasets, or other assets just because its module was imported.
A project can keep its own requirements.txt or environment file with pinned versions appropriate to its runtime.
🚀 A typical copy-and-adapt workflow
For a text classification project, a project might copy:

transformer_encoder_base/
├── data/datasets/text_classification/single_label.py
├── data/collators/classification_collator.py
├── models/attention/multi_head_attention.py
├── models/blocks/encoder_block.py
├── models/encoder/transformer_encoder.py
├── models/heads/classification_head.py
└── losses/classification_loss.py

common/
├── checkpoints/save.py
└── reproducibility/seed.py
Then adapt those files to the task's tokenizer, label schema, model dimensions, data splits, evaluation criteria, and runtime. This is an example of how the warehouse can be used; it does not imply that these exact files form a complete runnable project without integration work.
For a causal language model, a different project might copy decoder attention, positional encoding, a decoder block, the causal-LM collator, and generation helpers. For object detection, it might copy the annotation parser, box utilities, dataset adapter, and post-processing code.
🔬 Scratch implementations and reference code
Some components are intentionally educational or experimental. In particular, scratch RNN/LSTM/GRU and attention implementations can help inspect equations and tensor transformations. They are useful for learning, debugging, and modification, but should be benchmarked and reviewed before being selected for a performance-sensitive system.
Where a framework implementation is also available, the file name and description should make the distinction clear:
- Scratch/reference: implementation intended to expose the computation.
- Framework adapter: thin wrapper around a library implementation.
- Scaffold/template: starting interface that requires project-specific work.

⚠️ Scope and expectations
- This repository is a copy-and-adapt code warehouse, not a packaged library with a stable public API.
- Components may require additional files, dependencies, configuration, or project-specific glue.
- Export support depends on the chosen architecture, operators, framework versions, and target runtime. ONNX or TensorRT helpers do not guarantee that every model can be exported or accelerated.
- A training loop or model block does not guarantee a particular accuracy, speed, memory use, or production suitability.
- Validate copied code with the target dataset and environment; review preprocessing and label alignment carefully.
- Do not commit private data, credentials, large checkpoints, or machine-specific configuration unless the repository's policy explicitly allows them.

Possible future additions
The warehouse can grow as new reusable needs appear, for example:
- More dataset formats and task-specific collators.
- Additional evaluation and calibration metrics.
- Quantization and deployment examples for supported architectures.
- Robustness checks, profiling helpers, and data-validation snippets.
- Small end-to-end examples demonstrating how selected files connect.
New additions should preserve the copyable-file goal, lazy-import rule, and per-file English purpose comment.
