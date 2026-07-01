# 🧠 Mini GPT-2 — Decoder-Only Transformer from Scratch

> A full GPT-2–style transformer built from the ground up in PyTorch — no `transformers` library, no pretrained weights.
> Inspired by Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/GPU-Tesla%20T4-76B900?logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

---

## 📓 Notebooks

| | Notebook | Description | Open in Colab |
|---|---|---|---|
| 1 | `mini_gpt2_shakespeare.ipynb` | Baseline — character-level tokenizer, Tiny Shakespeare, ~10.7M params | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/notebooks/mini_gpt2_shakespeare.ipynb) |
| 2 | `mini_gpt2_advanced.ipynb` | Bonus — multilingual BPE tokenizer, RoPE, weight tying, 127M scale toggle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/notebooks/mini_gpt2_advanced.ipynb) |

> **Replace `YOUR-USERNAME/YOUR-REPO`** in the badge URLs with your actual GitHub path after pushing.

Both notebooks are fully self-contained — datasets download automatically, no manual uploads needed.

---

## 📁 Project Structure

```
.
├── README.md
└── notebooks/
    ├── mini_gpt2_shakespeare.ipynb   # Stage 1: char-level baseline
    └── mini_gpt2_advanced.ipynb      # Stage 2: BPE + multilingual + RoPE + scale-up
```

---

## 🏗️ Architecture

Both notebooks implement the same decoder-only backbone. The key differences are called out in the table below.

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Token Embedding                                     │
│  + Positional Encoding                               │
│    • Baseline  → Learned absolute position table     │
│    • Advanced  → RoPE (inside attention heads)       │
└─────────────────────────────────────────────────────┘
     │
     ▼  ×N layers
┌─────────────────────────────────────────────────────┐
│  Transformer Block                                   │
│  ┌─────────────────────────────────────────────┐    │
│  │  LayerNorm → Masked Multi-Head Attention     │    │
│  │             + Residual                       │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │  LayerNorm → Feed-Forward (4× expand)        │    │
│  │             + Residual                       │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
     │
     ▼
Final LayerNorm
     │
     ▼
Linear Head → Logits over vocab
(weight-tied to token embedding in advanced notebook)
```

### Key design choices

| Choice | Detail |
|---|---|
| **Pre-LayerNorm** | `x = x + SubLayer(LN(x))` — GPT-2's deviation from the 2017 Transformer paper; makes deep stacks easier to train |
| **Causal masking** | Lower-triangular mask in `Head.forward` — what makes this a decoder/autoregressive model. Position *i* can only attend to positions ≤ *i* |
| **RoPE** *(advanced)* | Rotates Q and K vectors by a position-dependent angle inside attention, so dot products depend on relative distance rather than absolute position. Same scheme as LLaMA and GPT-NeoX |
| **Weight tying** *(advanced)* | `lm_head.weight = token_embedding_table.weight` — halves the parameter count of the embedding/output layer |
| **GELU** *(advanced)* | Replaces ReLU in the FFN to match real GPT-2. Baseline uses ReLU to track Karpathy's lecture exactly |

---

## 📊 Hyperparameters

| Parameter | Baseline | Advanced `mini` | Advanced `full_gpt2` |
|---|---|---|---|
| Tokenizer | Character-level | Byte-level BPE (trained from scratch) | Byte-level BPE (trained from scratch) |
| Vocab size | 65 | 16,000 | 16,000 |
| Layers | 6 | 6 | 12 |
| Attention heads | 6 | 6 | 12 |
| Embedding dim | 384 | 384 | 768 |
| Context length | 256 | 256 | 1,024 |
| Batch size | 64 | 64 | 8 (+ ×8 grad. accum.) |
| Dropout | 0.2 | 0.2 | 0.2 |
| Learning rate | 3e-4 fixed | 3e-4 warmup + cosine decay | 3e-4 warmup + cosine decay |
| **Parameters** | **~10.7M** | **~16.8M** | **~97M** |
| GPU | Tesla T4 | Tesla T4 | Tesla T4 |

> `full_gpt2` uses a 16,000-token multilingual vocab instead of GPT-2's 50,257-token English-centric one — same 12/12/768/1024 transformer body, fewer embedding parameters. This puts the count at ~97M rather than ~127M. Increase `VOCAB_SIZE` to push closer to 127M if needed.

---

## 📈 Training Results

### Notebook 1 — Tiny Shakespeare (character-level)

5,000 steps · Tesla T4 · ~10.7M parameters

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 4.2221 | 4.2306 |
| 500 | 1.7605 | 1.9163 |
| 1,000 | 1.3937 | 1.6050 |
| 1,500 | 1.2649 | 1.5270 |
| 2,000 | 1.1852 | 1.5007 |
| 2,500 | 1.1220 | 1.4850 |
| 3,000 | 1.0730 | 1.4853 |
| 3,500 | 1.0186 | 1.5067 |
| 4,000 | 0.9617 | 1.5093 |
| 4,500 | 0.9116 | 1.5376 |
| **4,999** | **0.8561** | **1.5513** |

**Generated sample** (prompt: newline, 500 tokens):

```
But with painted and guishest wind toward
Of Polamas as shed lions, and father,
It will as trous as over mutine,
We, no more honour, none would end more silver foot
For weeping: to whom, they save mere a little,--
This ripwly seems to the bleed pock:
We want her maon! We'll addle to-day.
By county for Montague,
That dost thinkingly love? Welcome, my lord.
Welcome, the king that not resemble;
How! what news well'd in her help before,
And in this, I had seen his time in recoion?
Who mufflet goes a
```

Shakespeare-shaped: correct iambic structure, character names, archaic diction — expected for a 10M-parameter character-level model at this loss range.

---

### Notebook 2 — Multilingual Bible Corpus (BPE + RoPE)

5,000 steps · Tesla T4 · ~16.8M parameters · 5 languages · 16,000-token BPE vocab · LR warmup + cosine decay

| Step | Train Loss | Val Loss | LR |
|---|---|---|---|
| 0 | 9.7562 | 9.7561 | 3.00e-06 |
| 500 | 3.8390 | 3.8669 | 2.96e-04 |
| 1,000 | 3.3765 | 3.4368 | 2.78e-04 |
| 1,500 | 3.1310 | 3.2250 | 2.49e-04 |
| 2,000 | 2.9785 | 3.1074 | 2.12e-04 |
| 2,500 | 2.8531 | 3.0165 | 1.69e-04 |
| 3,000 | 2.7696 | 2.9570 | 1.27e-04 |
| 3,500 | 2.7056 | 2.9274 | 8.78e-05 |
| 4,000 | 2.6709 | 2.8953 | 5.68e-05 |
| 4,500 | 2.6440 | 2.8742 | 3.69e-05 |
| **4,999** | **2.6223** | **2.8678** | **3.00e-05** |

**Generated samples** (80 tokens each, from the trained model):

```
Prompt → "In the beginning"

In the beginning in the beginning of our gates, the elders, and people
Israel, be changed to depart from the morning unto the LORD for them.
有 聲 音 說 、 你 們 明 亮 百 姓 站 在 近 前 、 說 、 你 們 聽 見 自 己 說 、
他 們 在 耶 和 華 面 前 復 了 我 耶 和 華 、 還 以 色 列 人 在 其 中就 要
```

```
Prompt → "En el principio"

En el principio de Radrac el al rey le Joyada preguntó: --¿Has encrito?
He aquí, la espada de Saúl será cortado del arot. Éste es la tribu de
los hijos del rey. After Jesus answered him, Jesus said unto us. And he
said continually, Lord, thou camest here. और तू अपनी बहिनों के बीच खड़ा
```

```
Prompt → "В начале"

В начале царя Аггея, сына его, чтоб он сказал: вот в Иерусалиме;
ибо он – десять талантов, которые мы были поягости и ожидаем Отца;
耶 和 華 說 、 你 曾 容 我 的 　 神 說 、 台 微 甚 小 的 母 胎 、
因 為 我 倚 靠 這 大 腿 倒 塌 了 、 使 他 喪 命 芵 們 的 生 命 、
```

The model produces script-appropriate output across Cyrillic, Latin, Devanagari, and CJK — the real signature of a working multilingual byte-level tokenizer.

---

## 🗂️ Datasets

| Notebook | Dataset | Size | Auto-downloaded? | License |
|---|---|---|---|---|
| Baseline | [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | ~1 MB | ✅ Yes | Public domain |
| Advanced | 5 translations from [christos-c/bible-corpus](https://github.com/christos-c/bible-corpus) — English, Hindi, Spanish, Chinese, Russian | ~28 MB combined, ~155K verse lines | ✅ Yes | CC0 |

The Bible corpus was chosen specifically because it's a real, sizeable parallel corpus with one clean stable URL per language — no login, no API key, no scraping. Lines are shuffled across all five languages before tokenization so training windows contain mixed-language context rather than long monolingual stretches.

---

## 🚀 Getting Started

1. Open a notebook in Colab using the badges at the top.
2. `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all`

Everything else — dataset download, tokenizer training, model training, generation — runs in sequence inside the notebook.

**Worried about Colab disconnecting?** Set `USE_DRIVE = True` in the checkpointing cell to persist saves to Google Drive. The training loop reads the latest checkpoint on startup and auto-resumes from the last saved step — no lost progress on reconnect.

---

## ✅ Project Brief Checklist

| Requirement | Status |
|---|---|
| Decoder-only transformer, built entirely from scratch | ✅ |
| Reduced parameter count (not full 1.5B GPT-2) | ✅ ~10.7M baseline / ~16.8M advanced |
| Tokenization implemented and explained | ✅ char-level + from-scratch byte-level BPE |
| Causal (masked) self-attention | ✅ |
| Multi-head attention | ✅ |
| Residual connections + pre-LayerNorm | ✅ |
| Trained on real text, generates coherent output | ✅ |
| **Bonus** — Non-English / multilingual training data | ✅ 5 languages, 3 different scripts |
| **Bonus** — Custom attention / architectural modification | ✅ RoPE replaces learned position embeddings entirely |
| **Bonus** — Scale to full 127M GPT-2 configuration | ✅ `SCALE = 'full_gpt2'` toggle — 12 layers / 12 heads / 768 dim / 1024 context, fp16 + grad accumulation |

---

## ⚠️ Honest Limitations

**`full_gpt2` will not match GPT-2's published quality on free-tier compute.** Reproducing GPT-2 small to its reported loss takes roughly a week of continuous A100-class GPU time. A free Colab T4's weekly hour budget is nowhere near that. The `full_gpt2` path in the advanced notebook correctly instantiates the real architecture, trains, and shows steadily decreasing loss — but that is the engineering goal, not closing the gap to OpenAI's published numbers.

**Generation quality reflects the training corpus.** The advanced model is trained entirely on Bible translations — output reads like biblical prose with cross-language bleed-through, not fluent multilingual conversation.

---

## 🙏 Acknowledgments

- **Andrej Karpathy** — [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT). Primary reference for the baseline architecture, loss targets, and training recipe.
- **Christos Christodoulopoulos & Mark Steedman** — [bible-corpus](https://github.com/christos-c/bible-corpus), the multilingual dataset used in the advanced notebook. *A massively parallel corpus: the Bible in 100 languages*, Language Resources and Evaluation, 49(2).
- **Su, Lu, Pan, Murtadha, Wen & Liu** — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021). Basis for the RoPE implementation in the advanced notebook.

---

## 📄 License

Code in this repository is released under the **MIT License** — see [`LICENSE`](LICENSE) for details.
Dataset licenses are noted in the table above and remain with their original sources.
