# microGPT Visualizer

An interactive visual explainer for [Andrej Karpathy's microGPT](https://github.com/karpathy/microgpt): the complete GPT algorithm in 244 lines of Python with zero dependencies.

![Black theme, monospace, sidebar + 3 panels](https://img.shields.io/badge/style-dark%20minimal-000?style=flat-square)

## What It Does

Type a name → watch it flow through a transformer, step by step.

The visualizer breaks the GPT pipeline into three interactive stages:

| Panel | What You See |
|-------|-------------|
| **Tokenizer** | Characters mapped to integer IDs with BOS framing |
| **Embeddings** | Token + position vectors added together (16D heatmap) |
| **Attention** | Causal attention weights per head (4 heads × D_K=4) |

A scrollable sidebar shows reference material: architecture specs, autograd engine, MLP structure, training loop, and inference process.

## Accuracy

Every number is cross-referenced against `microgpt.py`:

- **4,064 parameters** (exact count: wte + wpe + attn + mlp + lm_head)
- **Hyperparameters**: n_embd=16, n_head=4, n_layer=1, block_size=8, vocab_size=27
- **Adam**: lr=0.01, β1=0.9, β2=0.95 (not 0.999), ε=1e-8, 500 steps
- **Activation**: ReLU² (not GeLU) : `xi.relu() ** 2`

## Run Locally

```bash
# Clone the repository
git clone https://github.com/Sjs2332/microGPT_Visualizer.git
cd microGPT_Visualizer

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Tech Stack

- **Next.js** + React + TypeScript
- **Tailwind CSS**: dark minimal theme
- Zero external ML libraries: all simulation is pure TypeScript math

## Project Structure

```
microgpt.py                    # The 244-line ground truth
src/
  app/page.tsx                 # Sidebar + 3 live panels
  components/visualizer/
    TokenizerViz.tsx           # Char → ID mapping
    EmbeddingViz.tsx           # Token + position embedding heatmap
    AttentionViz.tsx           # Causal attention matrix per head
  context/VisualizerContext.tsx  # Shared state (input → tokens → all panels)
  lib/microgpt-data.ts         # Config, tokenizer, embeddings, attention math
```

## License

MIT
