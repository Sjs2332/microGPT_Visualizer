"use client";

import { useState, useMemo } from "react";
import { useVisualizer } from "@/context/VisualizerContext";
import { generateEmbedding, generatePositionEmbedding, MODEL_CONFIG, BOS_TOKEN } from "@/lib/microgpt-data";

// Color scale: dark blue (negative/low) → cyan (zero) → yellow (positive/high)
function embeddingColor(value: number, maxVal: number): string {
    const t = (value / maxVal + 1) / 2; // normalize to 0..1 where 0.5 is zero
    if (t < 0.5) {
        // Negative: dark navy → cyan
        const s = t / 0.5;
        const r = Math.round(10 + s * 40);
        const g = Math.round(20 + s * 200);
        const b = Math.round(80 + s * 175);
        return `rgb(${r},${g},${b})`;
    } else {
        // Positive: cyan → amber/yellow
        const s = (t - 0.5) / 0.5;
        const r = Math.round(50 + s * 205);
        const g = Math.round(220 - s * 30);
        const b = Math.round(255 - s * 225);
        return `rgb(${r},${g},${b})`;
    }
}

function HeatmapRow({
    label,
    values,
    maxVal,
    dimLabel,
}: {
    label: string;
    values: number[];
    maxVal: number;
    dimLabel?: string;
}) {
    return (
        <div className="flex items-center gap-2">
            <span className="text-[8px] text-zinc-500 w-8 text-right uppercase shrink-0">{label}</span>
            <div className="flex gap-px flex-1">
                {values.map((v, i) => (
                    <div
                        key={i}
                        className="flex-1 h-5 relative group cursor-default"
                        style={{ backgroundColor: embeddingColor(v, maxVal) }}
                    >
                        <span className="opacity-0 group-hover:opacity-100 absolute -top-7 left-1/2 -translate-x-1/2 bg-zinc-900 text-white px-1.5 py-0.5 text-[7px] whitespace-nowrap z-50 border border-white/20">
                            [{i}] {v.toFixed(3)}
                        </span>
                    </div>
                ))}
            </div>
            {dimLabel && <span className="text-[7px] text-zinc-700 w-6 shrink-0">{dimLabel}</span>}
        </div>
    );
}

// Color legend bar
function EmbeddingLegend() {
    return (
        <div className="flex items-center gap-2 text-[7px] text-zinc-500">
            <span>−</span>
            <div className="flex-1 h-2.5 rounded-sm overflow-hidden flex">
                {Array.from({ length: 40 }).map((_, i) => {
                    const t = (i / 39) * 2 - 1; // -1 to 1
                    return (
                        <div
                            key={i}
                            className="flex-1 h-full"
                            style={{ backgroundColor: embeddingColor(t, 1) }}
                        />
                    );
                })}
            </div>
            <span>+</span>
        </div>
    );
}

export function EmbeddingViz() {
    const { tokens, tokenLabels, vocab } = useVisualizer();
    const dim = MODEL_CONFIG.n_embd;

    const [selectedIdx, setSelectedIdx] = useState(1);

    const maxIdx = tokens ? tokens.length - 1 : 0;
    const idx = Math.min(selectedIdx, maxIdx);
    const tokenId = tokens ? tokens[idx] : 5;
    const posId = idx;
    const charLabel = tokenLabels[idx] || "?";

    const tokEmb = useMemo(() => generateEmbedding(dim, tokenId), [dim, tokenId]);
    const posEmb = useMemo(() => generatePositionEmbedding(posId, dim), [dim, posId]);
    const combined = useMemo(() => tokEmb.map((t, i) => t + posEmb[i]), [tokEmb, posEmb]);

    const maxVal = Math.max(
        ...tokEmb.map(Math.abs),
        ...posEmb.map(Math.abs),
        ...combined.map(Math.abs),
        0.01
    );

    return (
        <div className="space-y-4 font-mono text-white">
            <div className="border-b border-white pb-2">
                <h2 className="text-sm font-bold uppercase tracking-widest">2. Embeddings</h2>
                <p className="text-[9px] text-zinc-500 mt-1">
                    Each token ID becomes a {dim}-dimensional vector. Position gets its own vector.
                    They&apos;re added together.
                </p>
            </div>

            {/* Token selector */}
            {tokens && tokens.length > 0 && (
                <div className="space-y-1">
                    <p className="text-[8px] text-zinc-500 uppercase">Select token to inspect</p>
                    <div className="flex flex-wrap gap-1">
                        {tokens.map((t, i) => {
                            const ch = vocab.itos[t];
                            const isBos = ch === BOS_TOKEN;
                            return (
                                <button
                                    key={i}
                                    onClick={() => setSelectedIdx(i)}
                                    className={`px-2 py-1 text-[9px] border transition-colors ${idx === i
                                            ? "bg-white text-black border-white font-bold"
                                            : isBos
                                                ? "border-white/20 text-zinc-600 hover:border-white/40"
                                                : "border-white/30 text-zinc-400 hover:border-white/50"
                                        }`}
                                >
                                    {isBos ? "BOS" : ch}
                                    <span className="text-[7px] ml-0.5 opacity-50">#{t}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Embedding vis */}
            <div className="space-y-2">
                <p className="text-[8px] text-zinc-500 uppercase">
                    &quot;{charLabel}&quot; (ID={tokenId}) at position {posId}
                </p>

                <HeatmapRow label="Tok" values={tokEmb} maxVal={maxVal} dimLabel={`${dim}D`} />
                <div className="text-center text-[8px] text-zinc-600">+</div>
                <HeatmapRow label="Pos" values={posEmb} maxVal={maxVal} dimLabel={`${dim}D`} />
                <div className="text-center text-[8px] text-zinc-600">═</div>
                <HeatmapRow label="Sum" values={combined} maxVal={maxVal} dimLabel={`${dim}D`} />
            </div>

            {/* Color legend */}
            <div className="space-y-1 border-t border-white/10 pt-2">
                <EmbeddingLegend />
                <p className="text-[8px] text-zinc-600">
                    <span className="text-cyan-400">Cool</span> = negative.
                    <span className="text-amber-300 ml-1">Warm</span> = positive.
                    Hover cells for exact values.
                </p>
                <p className="text-[7px] text-zinc-700">
                    X = WTE[{tokenId}] + WPE[{posId}] → {dim}D vector enters the transformer.
                </p>
            </div>
        </div>
    );
}
