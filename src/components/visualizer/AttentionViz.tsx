"use client";

import { useState, useMemo } from "react";
import { useVisualizer } from "@/context/VisualizerContext";
import { computeAttentionWeights, MODEL_CONFIG, BOS_TOKEN } from "@/lib/microgpt-data";

// Attention color: transparent/dark → vibrant teal/green for high attention
function attentionColor(weight: number, maxWeight: number): string {
    const t = maxWeight > 0 ? weight / maxWeight : 0; // 0..1
    // Dark → teal → bright green
    const r = Math.round(10 + t * 30);
    const g = Math.round(20 + t * 230);
    const b = Math.round(40 + t * 140);
    const a = (0.15 + t * 0.85).toFixed(2);
    return `rgba(${r},${g},${b},${a})`;
}

export function AttentionViz() {
    const { tokens, tokenLabels, vocab } = useVisualizer();
    const numHeads = MODEL_CONFIG.n_head;
    const dK = MODEL_CONFIG.n_embd / numHeads;

    const [head, setHead] = useState(0);

    const seqLen = tokens ? tokens.length : 6;
    const labels = tokens
        ? tokens.map((t) => {
            const ch = vocab.itos[t];
            return ch === BOS_TOKEN ? "BOS" : ch;
        })
        : ["BOS", "e", "m", "m", "a", "BOS"];

    const weights = useMemo(
        () => computeAttentionWeights(seqLen, head),
        [seqLen, head]
    );

    const maxW = Math.max(...weights.flat(), 0.01);

    return (
        <div className="space-y-4 font-mono text-white">
            <div className="border-b border-white pb-2">
                <h2 className="text-sm font-bold uppercase tracking-widest">3. Attention</h2>
                <p className="text-[9px] text-zinc-500 mt-1">
                    Each token decides how much to &quot;look at&quot; previous tokens.
                    The model has {numHeads} heads, each seeing {dK} dimensions.
                </p>
            </div>

            {/* Head selector */}
            <div className="space-y-1">
                <p className="text-[8px] text-zinc-500 uppercase">Attention head</p>
                <div className="flex gap-1">
                    {Array.from({ length: numHeads }).map((_, h) => (
                        <button
                            key={h}
                            onClick={() => setHead(h)}
                            className={`px-2.5 py-1 text-[9px] border transition-colors ${head === h
                                    ? "bg-white text-black border-white font-bold"
                                    : "border-white/30 text-zinc-400 hover:border-white/50"
                                }`}
                        >
                            H{h}
                        </button>
                    ))}
                    <span className="text-[8px] text-zinc-600 self-center ml-1">
                        D<sub>K</sub>={dK}
                    </span>
                </div>
            </div>

            {/* Attention matrix */}
            <div className="space-y-1">
                <div className="flex items-center gap-2 text-[8px] text-zinc-500 uppercase mb-1">
                    <span>↓ queries</span>
                    <span className="text-zinc-700">|</span>
                    <span>→ keys</span>
                </div>

                <div className="inline-block">
                    {/* Column headers */}
                    <div className="flex">
                        <div className="w-9" />
                        {labels.map((l, j) => (
                            <div
                                key={`col-${j}`}
                                className="w-9 h-5 flex items-end justify-center text-[7px] text-zinc-500"
                            >
                                {l}
                            </div>
                        ))}
                    </div>

                    {/* Rows */}
                    {weights.map((row: number[], i: number) => (
                        <div key={`row-${i}`} className="flex">
                            <div className="w-9 h-8 flex items-center justify-end pr-1 text-[7px] text-zinc-500">
                                {labels[i]}
                            </div>
                            {row.map((w: number, j: number) => {
                                const isMasked = j > i;
                                return (
                                    <div
                                        key={`${i}-${j}`}
                                        className={`w-9 h-8 border border-white/5 flex items-center justify-center text-[7px] relative group cursor-default ${isMasked ? "opacity-[0.04]" : ""
                                            }`}
                                        style={{
                                            backgroundColor: isMasked ? "transparent" : attentionColor(w, maxW),
                                            color: w / maxW > 0.5 ? "black" : "rgba(255,255,255,0.8)",
                                        }}
                                    >
                                        {!isMasked && (
                                            <>
                                                <span className="font-bold">{Math.round(w * 100)}</span>
                                                <span className="opacity-0 group-hover:opacity-100 absolute -top-6 left-1/2 -translate-x-1/2 bg-zinc-900 text-white px-1.5 py-0.5 text-[7px] whitespace-nowrap z-50 border border-white/20">
                                                    {labels[i]}→{labels[j]}: {(w * 100).toFixed(1)}%
                                                </span>
                                            </>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    ))}
                </div>
            </div>

            {/* Color legend */}
            <div className="space-y-1 border-t border-white/10 pt-2">
                <div className="flex items-center gap-2 text-[7px] text-zinc-500">
                    <span>0%</span>
                    <div className="flex-1 h-2.5 rounded-sm overflow-hidden flex">
                        {Array.from({ length: 30 }).map((_, i) => (
                            <div
                                key={i}
                                className="flex-1 h-full"
                                style={{ backgroundColor: attentionColor(i / 29, 1) }}
                            />
                        ))}
                    </div>
                    <span>100%</span>
                </div>
                <p className="text-[8px] text-zinc-600">
                    <span style={{ color: "rgb(40,250,180)" }}>Bright</span> = high attention.
                    Numbers = % of attention each query gives to each key.
                    Greyed cells = <span className="text-zinc-500">masked</span> (can&apos;t see future).
                </p>
                <p className="text-[7px] text-zinc-700">
                    CAUSAL: pos[i] sees only pos[0..i]. Formula: softmax(Q·K<sup>T</sup> / √{dK})
                </p>
            </div>
        </div>
    );
}
