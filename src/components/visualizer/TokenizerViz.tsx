"use client";

import { useVisualizer } from "@/context/VisualizerContext";
import { BOS_TOKEN, MODEL_CONFIG } from "@/lib/microgpt-data";

export function TokenizerViz() {
    const { inputText, setInputText, vocab, tokens } = useVisualizer();

    return (
        <div className="space-y-4 font-mono text-white">
            <div className="border-b border-white pb-2">
                <h2 className="text-sm font-bold uppercase tracking-widest">1. Tokenizer</h2>
                <p className="text-[9px] text-zinc-500 mt-1">
                    Converts each character into an integer the model can process.
                </p>
            </div>

            {/* Input */}
            <div className="space-y-1">
                <label className="text-[8px] text-zinc-500 uppercase">Input name</label>
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    className="w-full bg-black border border-white/40 p-2.5 text-sm uppercase tracking-wider focus:outline-none focus:border-white transition-colors"
                    placeholder="TYPE A NAME..."
                    maxLength={7}
                />
            </div>

            {/* Token display */}
            {tokens ? (
                <div className="space-y-3">
                    {/* Character → ID mapping */}
                    <div>
                        <p className="text-[8px] text-zinc-500 uppercase mb-1.5">
                            Character → Token ID
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                            {tokens.map((t, i) => {
                                const ch = vocab.itos[t];
                                const isBos = ch === BOS_TOKEN;
                                return (
                                    <div key={i} className="flex flex-col items-center">
                                        {/* Character */}
                                        <div
                                            className={`px-2 py-1 text-xs font-bold border ${isBos
                                                ? "border-white/30 text-zinc-500 bg-zinc-900"
                                                : "border-white/50 bg-black"
                                                }`}
                                        >
                                            {isBos ? "BOS" : ch}
                                        </div>
                                        {/* Arrow */}
                                        <div className="text-[8px] text-zinc-600 my-0.5">↓</div>
                                        {/* Integer */}
                                        <div className="text-[10px] font-bold text-zinc-400">{t}</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Resulting sequence */}
                    <div className="border border-white/10 p-2.5">
                        <p className="text-[8px] text-zinc-500 uppercase mb-1">
                            Token sequence (fed to model)
                        </p>
                        <p className="text-xs font-bold tracking-wider">
                            [{tokens.join(", ")}]
                        </p>
                        <p className="text-[8px] text-zinc-600 mt-1.5">
                            Length: {tokens.length} tokens | Vocab: {vocab.vocabSize} chars (full dataset: {MODEL_CONFIG.vocab_size})
                        </p>
                    </div>

                    {/* Explanation */}
                    <div className="text-[8px] text-zinc-600 leading-relaxed space-y-1 border-t border-white/10 pt-2">
                        <p>
                            <span className="text-zinc-400 font-bold">BOS</span> = Beginning of Sequence.
                            The leading BOS tells the model &quot;start here.&quot;
                            The trailing BOS means &quot;end of name.&quot;
                        </p>
                        <p className="text-zinc-700">
                            microgpt.py line 200: tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
                        </p>
                    </div>
                </div>
            ) : (
                <div className="text-[10px] text-zinc-600 uppercase border border-white/10 p-3 text-center">
                    {inputText ? "INVALID : ONLY a-z CHARACTERS" : "AWAITING INPUT..."}
                </div>
            )}
        </div>
    );
}
