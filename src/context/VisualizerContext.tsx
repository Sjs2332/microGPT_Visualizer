"use client";

import { createContext, useContext, useState, useMemo, type ReactNode } from "react";
import { buildVocab, tokenize, SAMPLE_NAMES } from "@/lib/microgpt-data";

// ─── Shared State ────────────────────────────────────────────────────────────
// This is the "wire" that connects all 8 panels. When the user types a name
// in the Tokenizer, every downstream panel reacts to the same token sequence.

interface VisualizerState {
    // Input
    inputText: string;
    setInputText: (text: string) => void;
    // Derived
    vocab: ReturnType<typeof buildVocab>;
    tokens: number[] | null;       // null if input has invalid chars
    tokenLabels: string[];         // human-readable labels for the token sequence
}

const VisualizerContext = createContext<VisualizerState | null>(null);

export function VisualizerProvider({ children }: { children: ReactNode }) {
    const [inputText, setInputText] = useState("emma");

    // Build vocab once from the sample dataset (microgpt.py lines 26-29)
    const vocab = useMemo(() => buildVocab(SAMPLE_NAMES), []);

    // Tokenize the current input: this feeds Embeddings, Attention, and Inference
    const tokens = useMemo(() => {
        if (!inputText) return null;
        const valid = [...inputText.toLowerCase()].every(c => c in vocab.stoi);
        if (!valid) return null;
        return tokenize(inputText, vocab.stoi);
    }, [inputText, vocab]);

    // Human-readable labels for the heatmap/attention axes
    const tokenLabels = useMemo(() => {
        if (!tokens) return [];
        return tokens.map(t => {
            const ch = vocab.itos[t];
            return ch === "<BOS>" ? "BOS" : ch;
        });
    }, [tokens, vocab]);

    return (
        <VisualizerContext.Provider
            value={{ inputText, setInputText, vocab, tokens, tokenLabels }}
        >
            {children}
        </VisualizerContext.Provider>
    );
}

export function useVisualizer(): VisualizerState {
    const ctx = useContext(VisualizerContext);
    if (!ctx) throw new Error("useVisualizer must be used within VisualizerProvider");
    return ctx;
}
