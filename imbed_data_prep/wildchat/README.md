# WildChat

Data preparation for the WildChat dataset.

## Data source

[WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) is a
collection of one million real-world conversations between human users and
ChatGPT, collected by AllenAI by offering free access to GPT-3.5 and GPT-4.

- **25.53%** of conversations come from GPT-4; the rest from GPT-3.5
- Includes **demographic data:** state, country, hashed IP, request headers
- Covers **diverse interactions:** ambiguous requests, code-switching,
  topic-switching, political discussions
- **De-identified** with Microsoft Presidio and hand-written rules

The default `WildChat-1M` version contains only non-toxic conversations.
For toxic data, see
[WildChat-1M-Full](https://huggingface.co/datasets/allenai/WildChat-1M-Full)
(requires approval).

- **License:** ODC-BY
- **Paper:** [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470)

## What it does

1. **Load from Hugging Face** -- download via the `datasets` library.
2. **Expand conversations** -- flatten nested conversation and moderation
   arrays into individual rows.
3. **Filter to English** -- keep only English-language messages.
4. **Prepare embeddable segments** -- add token counts and validate by
   conversation hash.

## Output

A DataFrame with columns:

| Column | Description |
|---|---|
| `conversation_id` | Unique conversation identifier |
| `conversation.content` | Message text |
| `conversation.role` | `user` or `assistant` |
| `language` | Detected language |
| `model` | ChatGPT model used |
| `turn` | Turn index within conversation |
| `toxic` | Toxicity flag from moderation |
| `redacted` | Whether PII was redacted |

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Module code |
| `wildchat.ipynb` | Notebook exploring the WildChat data |
