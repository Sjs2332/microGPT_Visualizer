// ─── Model Configuration (from microgpt.py lines 118-122) ────────────────────
// These MUST match the original source exactly.
export const MODEL_CONFIG = {
    n_embd: 16,       // embedding dimension
    n_head: 4,        // number of attention heads
    n_layer: 1,       // number of layers
    block_size: 8,    // maximum sequence length (NOT 16)
    vocab_size: 27,   // 26 lowercase letters + <BOS>
    head_dim: 4,      // n_embd / n_head
};

// ─── Optimizer Configuration (from microgpt.py line 190) ─────────────────────
export const OPTIMIZER_CONFIG = {
    learning_rate: 1e-2,  // 0.01 (NOT 0.001)
    beta1: 0.9,
    beta2: 0.95,          // (NOT 0.999 - Karpathy uses 0.95)
    eps: 1e-8,
    num_steps: 500,
};

export const BOS_TOKEN = "<BOS>";
export const SAMPLE_NAMES = [
    "emma", "olivia", "ava", "isabella", "sophia",
    "mia", "charlotte", "amelia", "evelyn", "abigail",
];

// ─── Parameter Count (matches microgpt.py line 132-133) ──────────────────────
export function countParameters() {
    const { n_embd, n_head, n_layer, block_size, vocab_size } = MODEL_CONFIG;
    const wte = vocab_size * n_embd;           // token embedding table
    const wpe = block_size * n_embd;           // position embedding table
    // per layer: attn_wq, attn_wk, attn_wv (3 * n_embd * n_embd) + attn_wo (n_embd * n_embd)
    const attn = n_layer * (n_embd * n_embd * 4);
    // per layer: mlp_fc1 (4*n_embd * n_embd) + mlp_fc2 (n_embd * 4*n_embd)
    const mlp = n_layer * (n_embd * 4 * n_embd * 2);
    const lm_head = vocab_size * n_embd;      // output projection
    return wte + wpe + attn + mlp + lm_head;
}

export const PIPELINE_STEPS = [
    { id: "tokenizer", label: "TOKENIZER", formula: "STR → INT[]" },
    { id: "embedding", label: "EMBEDDING", formula: "INT → VEC(16)" },
    { id: "rmsnorm", label: "RMSNORM", formula: "X / √(MEAN(X²))" },
    { id: "attention", label: "ATTENTION", formula: "SOFTMAX(QK^T/√d)V" },
    { id: "mlp", label: "MLP", formula: "FC2(RELU²(FC1(X)))" },
    { id: "lm_head", label: "LM_HEAD", formula: "VEC(16) → LOGITS(27)" },
    { id: "softmax", label: "SOFTMAX", formula: "LOGITS → PROBS" },
    { id: "sample", label: "SAMPLE", formula: "PROBS → NEXT_TOKEN" },
];

// ─── Tokenizer (from microgpt.py lines 26-30) ───────────────────────────────
export function buildVocab(docs: string[]) {
    const chars = [BOS_TOKEN, ...Array.from(new Set(docs.join(""))).sort()];
    const stoi: Record<string, number> = {};
    const itos: Record<number, string> = {};
    chars.forEach((ch, i) => {
        stoi[ch] = i;
        itos[i] = ch;
    });
    return { chars, stoi, itos, vocabSize: chars.length };
}

export function tokenize(text: string, stoi: Record<string, number>): number[] {
    const tokens = [stoi[BOS_TOKEN]];
    for (const char of text.toLowerCase()) {
        if (char in stoi) tokens.push(stoi[char]);
    }
    tokens.push(stoi[BOS_TOKEN]); // trailing BOS as in microgpt.py line 200
    return tokens;
}

// ─── Embeddings (from microgpt.py lines 152-155) ─────────────────────────────
// Simulated: we use deterministic sin/cos to mimic learned embeddings.
export function generateEmbedding(dim: number, seed: number): number[] {
    return Array.from({ length: dim }, (_, i) =>
        parseFloat((Math.sin(seed * 1.37 + i * 0.73) * 0.5).toFixed(4))
    );
}

export function generatePositionEmbedding(pos: number, dim: number): number[] {
    return Array.from({ length: dim }, (_, i) =>
        parseFloat((Math.cos(pos * 2.17 + i * 0.91) * 0.3).toFixed(4))
    );
}

// ─── Autograd (from microgpt.py lines 35-115) ────────────────────────────────
// Simplified computation graph: ReLU(a*b + c)
export interface ValueNode {
    id: string;
    label: string;
    data: number;
    grad: number;
    op: string;
    children: string[]; // IDs of input nodes
}

export function buildComputationGraph(): ValueNode[] {
    // Forward: a=2, b=-3, c=10 → a*b = -6 → (-6)+10 = 4 → ReLU(4) = 4
    // Backward: dout/dout = 1, ReLU grad = 1 (since 4>0), d(sum)/d(a*b) = 1,
    //   d(sum)/dc = 1, d(a*b)/da = b = -3, d(a*b)/db = a = 2
    return [
        { id: "a", label: "a", data: 2.0, grad: -3.0, op: "", children: [] },
        { id: "b", label: "b", data: -3.0, grad: 2.0, op: "", children: [] },
        { id: "c", label: "c", data: 10.0, grad: 1.0, op: "", children: [] },
        { id: "ab", label: "a*b", data: -6.0, grad: 1.0, op: "*", children: ["a", "b"] },
        { id: "sum", label: "a*b+c", data: 4.0, grad: 1.0, op: "+", children: ["ab", "c"] },
        { id: "out", label: "out", data: 4.0, grad: 1.0, op: "ReLU", children: ["sum"] },
    ];
}

// ─── Attention (from microgpt.py lines 158-175) ──────────────────────────────
// Computes causal attention weights with softmax.
export function computeAttentionWeights(seqLen: number, headSeed: number): number[][] {
    const weights: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
        const row: number[] = [];
        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
            if (j > i) {
                row.push(0); // causal mask: can't attend to future
            } else {
                const val = Math.exp(Math.sin(headSeed + i * 13 + j * 7) * 2);
                row.push(val);
                sum += val;
            }
        }
        // softmax normalization over valid positions
        weights.push(row.map(v => (sum > 0 ? v / sum : 0)));
    }
    return weights;
}

// ─── Activation Functions (from microgpt.py line 182) ────────────────────────
export const relu = (x: number): number => Math.max(0, x);
export const reluSquared = (x: number): number => Math.pow(Math.max(0, x), 2);

export function generateActivationData(
    fn: (x: number) => number,
    range: [number, number],
    steps: number
): { x: number; y: number }[] {
    const data: { x: number; y: number }[] = [];
    const dx = (range[1] - range[0]) / steps;
    for (let i = 0; i <= steps; i++) {
        const x = range[0] + i * dx;
        data.push({ x, y: fn(x) });
    }
    return data;
}

// ─── Training Loss Simulation ────────────────────────────────────────────────
// Simulates the loss curve shape from microgpt.py's 500-step training loop.
export function generateTrainingLoss(numSteps: number): { step: number; loss: number }[] {
    const data: { step: number; loss: number }[] = [];
    // Seed for reproducibility within the visualization
    let loss = 3.3; // ~ln(27) ≈ 3.3 is the random-guess loss for vocab_size=27
    for (let i = 0; i < numSteps; i++) {
        const progress = i / numSteps;
        // Exponential decay with noise, mimicking real training
        const target = 3.3 * Math.exp(-3.5 * progress) + 0.3;
        loss = loss * 0.95 + target * 0.05 + (Math.sin(i * 0.7) * 0.03);
        data.push({ step: i, loss: Math.max(0.2, loss) });
    }
    return data;
}

// ─── Inference Probability Simulation ────────────────────────────────────────
// Simulates softmax(logits / temperature) for a set of characters.
export function simulateNextTokenProbs(
    temperature: number,
    vocabChars: string[]
): { char: string; prob: number }[] {
    // Simulate some learned logits (deterministic per char)
    const logits = vocabChars.map((_, i) => Math.sin(i * 1.7 + 0.3) * 2);
    // Temperature scaling (from microgpt.py line 238)
    const scaled = logits.map(l => l / temperature);
    const maxScaled = Math.max(...scaled);
    const exps = scaled.map(l => Math.exp(l - maxScaled)); // numerically stable
    const sum = exps.reduce((a, b) => a + b, 0);
    return vocabChars.map((char, i) => ({
        char: char === BOS_TOKEN ? "⏎" : char,
        prob: exps[i] / sum,
    }));
}

// ─── Inference Sampling ──────────────────────────────────────────────────────
// Replicates microgpt.py lines 232-243: sample until BOS or max length.
export function sampleName(
    temperature: number,
    vocab: ReturnType<typeof buildVocab>
): string {
    let result = "";
    const maxLen = MODEL_CONFIG.block_size;
    for (let pos = 0; pos < maxLen; pos++) {
        // Simulate logits based on what we've generated so far
        const contextSeed = result.length > 0
            ? result.charCodeAt(result.length - 1) + pos
            : 42 + pos;
        const logits = vocab.chars.map((_, i) =>
            Math.sin(contextSeed * 0.37 + i * 1.13 + pos * 0.7) * 2
        );
        // Temperature scaling + softmax
        const scaled = logits.map(l => l / temperature);
        const maxS = Math.max(...scaled);
        const exps = scaled.map(l => Math.exp(l - maxS));
        const sum = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sum);
        // Weighted random sampling (like random.choices in Python)
        let r = Math.random();
        let tokenId = 0;
        for (let i = 0; i < probs.length; i++) {
            r -= probs[i];
            if (r <= 0) { tokenId = i; break; }
        }
        // Stop at BOS (microgpt.py line 240-241)
        if (tokenId === 0) break;
        result += vocab.itos[tokenId];
    }
    return result;
}
