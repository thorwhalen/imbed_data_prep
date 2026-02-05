# LMSYS AI Conversations

Data preparation for the LMSYS-Chat-1M dataset.

## Data source

[LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) is a
large-scale dataset of one million real-world conversations with 25
state-of-the-art LLMs, collected from 210K unique IP addresses via the
Vicuna demo and Chatbot Arena platforms.

- **Paper:** [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation
  Dataset](https://arxiv.org/abs/2309.11998) (Zheng et al., 2023)
- **Hugging Face:** `lmsys/lmsys-chat-1m`
- **License:** LMSYS-Chat-1M Dataset License Agreement (research and
  commercial use permitted)

The dataset contains raw conversation text with OpenAI Moderation API
outputs.  PII has been removed.

## What it does

1. **Load from Hugging Face** -- download the dataset via the `datasets`
   library.
2. **Flatten conversations** -- expand the nested conversation and moderation
   arrays into individual rows, one per message turn.
3. **Filter to English** -- keep only English-language conversations.
4. **Filter by token count** -- remove messages that are too short or too
   long for embedding.
5. **Compute embeddings** -- batch-compute OpenAI embeddings with chunked
   saves.
6. **Project to 2D** -- UMAP projection of the embedding space.
7. **Incremental PCA** -- optional PCA with incremental fitting for large
   datasets.

## Output

A DataFrame with columns:

| Column | Description |
|---|---|
| `conversation_id` | Unique conversation identifier |
| `content` | Message text |
| `role` | `user` or `assistant` |
| `language` | Detected language |
| `model` | LLM model that generated the response |
| `num_of_tokens` | Token count |
| `flagged` | OpenAI moderation flag |
| `categories` | Moderation categories triggered |
| `category_scores` | Moderation category scores |
| embedding columns | High-dimensional embedding vector |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
