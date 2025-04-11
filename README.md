# 🧠 TinyGPT: 7-Parameter GPT Model

This is a deliberately minimal GPT-style transformer with **only 7 trainable parameters**.

Why? Because we can.

## Architecture

- 1-dim token embedding (2 vocab tokens)
- 1 position embedding
- Linear output projection

## Usage

```bash
pip install -r requirements.txt
python train.py
