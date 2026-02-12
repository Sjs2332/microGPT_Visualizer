"use client";

import { useEffect, useState, useMemo } from "react";
import { VisualizerProvider } from "@/context/VisualizerContext";
import { TokenizerViz } from "@/components/visualizer/TokenizerViz";
import { EmbeddingViz } from "@/components/visualizer/EmbeddingViz";
import { AttentionViz } from "@/components/visualizer/AttentionViz";
import {
  MODEL_CONFIG,
  OPTIMIZER_CONFIG,
  PIPELINE_STEPS,
  countParameters,
  generateActivationData,
  generateTrainingLoss,
  relu,
  reluSquared,
} from "@/lib/microgpt-data";

// ─── Inline SVG chart ────────────────────────────────────────────────────────

function MiniChart({ data, w = 160, h = 35 }: { data: { x: number; y: number }[]; w?: number; h?: number }) {
  const pad = 4;
  const maxY = Math.max(...data.map((d) => d.y), 0.1);
  const minX = data[0].x;
  const maxX = data[data.length - 1].x;
  const toX = (x: number) => pad + ((x - minX) / (maxX - minX)) * (w - pad * 2);
  const toY = (y: number) => h - pad - (y / maxY) * (h - pad * 2);
  const pathD = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${toX(d.x).toFixed(1)} ${toY(d.y).toFixed(1)}`)
    .join(" ");
  return (
    <svg width={w} height={h} className="border border-white/10 bg-black w-full">
      <path d={pathD} fill="none" stroke="white" strokeWidth="1" />
    </svg>
  );
}

// ─── Sidebar section divider ─────────────────────────────────────────────────

function SideSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="py-3">
      <p className="text-[9px] font-bold uppercase tracking-widest text-zinc-400 border-b border-white/10 pb-1.5 mb-2">
        {title}
      </p>
      <div className="text-[9px] text-zinc-300 leading-relaxed space-y-1.5">{children}</div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function Home() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const totalParams = countParameters();
  const reluData = useMemo(() => generateActivationData(relu, [-2, 2], 40), []);
  const reluSqData = useMemo(() => generateActivationData(reluSquared, [-2, 2], 40), []);
  const lossData = useMemo(() => generateTrainingLoss(OPTIMIZER_CONFIG.num_steps), []);

  const lossW = 180;
  const lossH = 40;
  const lossPad = 6;
  const maxL = Math.max(...lossData.map((d) => d.loss));
  const minL = Math.min(...lossData.map((d) => d.loss));
  const lossToX = (s: number) => parseFloat((lossPad + (s / OPTIMIZER_CONFIG.num_steps) * (lossW - lossPad * 2)).toFixed(1));
  const lossToY = (l: number) => parseFloat((lossH - lossPad - ((l - minL) / (maxL - minL + 0.01)) * (lossH - lossPad * 2)).toFixed(1));
  const lossPath = lossData.map((d, i) => `${i === 0 ? "M" : "L"} ${lossToX(d.step)} ${lossToY(d.loss)}`).join(" ");

  if (!mounted) return <div className="h-screen w-screen bg-black" />;

  return (
    <VisualizerProvider>
      <div className="h-screen w-screen overflow-hidden bg-black text-white font-mono flex">
        {/* ── Left Sidebar: All reference content inline ── */}
        <div className="w-56 shrink-0 border-r border-white/10 overflow-y-auto px-3 py-2">
          <p className="text-[8px] text-zinc-600 uppercase tracking-[0.2em] mb-1">
            microGPT Reference
          </p>

          {/* Architecture */}
          <SideSection title="Architecture">
            <div className="grid grid-cols-3 gap-1">
              {[
                { l: "EMBD", v: MODEL_CONFIG.n_embd },
                { l: "HEADS", v: MODEL_CONFIG.n_head },
                { l: "LAYERS", v: MODEL_CONFIG.n_layer },
                { l: "BLK_SZ", v: MODEL_CONFIG.block_size },
                { l: "VOCAB", v: MODEL_CONFIG.vocab_size },
                { l: "PARAMS", v: totalParams.toLocaleString() },
              ].map(({ l, v }) => (
                <div key={l} className="border border-white/10 p-1">
                  <p className="text-[7px] text-zinc-600">{l}</p>
                  <p className="font-bold text-[10px]">{v}</p>
                </div>
              ))}
            </div>
            <div className="space-y-0.5 mt-2">
              {PIPELINE_STEPS.map((s, i) => (
                <div key={s.id} className="flex gap-1 text-[8px]">
                  <span className="text-zinc-600 w-2.5">{i + 1}</span>
                  <span className="font-bold w-14 text-zinc-300">{s.label}</span>
                  <span className="text-zinc-600 truncate">{s.formula}</span>
                </div>
              ))}
            </div>
          </SideSection>

          {/* Autograd */}
          <SideSection title="Autograd">
            <p className="text-zinc-500 text-[8px]">
              Every number is a Value(data, grad). Operations build a graph.
              Backward walks it in reverse, applying the chain rule.
            </p>
            <div className="border border-white/10 p-1.5 space-y-0.5 text-[8px]">
              <p className="font-bold">out = ReLU(a·b + c)</p>
              <p className="text-zinc-500">a=2, b=-3, c=10</p>
              <p className="text-zinc-500">FWD: 2·(-3)=-6 → -6+10=4 → ReLU(4)=4</p>
              <p className="text-zinc-500">BWD: ∂out/∂a = -3, ∂out/∂b = 2, ∂out/∂c = 1</p>
            </div>
          </SideSection>

          {/* MLP */}
          <SideSection title="MLP (Feed-Forward)">
            <div className="space-y-0.5 text-[8px]">
              <p>FC1: {MODEL_CONFIG.n_embd}D → {MODEL_CONFIG.n_embd * 4}D</p>
              <p className="font-bold">ACT: RELU² (not GeLU)</p>
              <p>FC2: {MODEL_CONFIG.n_embd * 4}D → {MODEL_CONFIG.n_embd}D</p>
              <p className="text-zinc-500">+ residual skip connection</p>
            </div>
            <div className="flex gap-1 mt-1">
              <div className="flex-1">
                <p className="text-[7px] text-zinc-600 mb-0.5">RELU</p>
                <MiniChart data={reluData} h={30} />
              </div>
              <div className="flex-1">
                <p className="text-[7px] text-zinc-600 mb-0.5">RELU²</p>
                <MiniChart data={reluSqData} h={30} />
              </div>
            </div>
          </SideSection>

          {/* Training */}
          <SideSection title="Training">
            <div className="text-[8px] space-y-0.5">
              <div className="flex justify-between text-zinc-500">
                <span>{OPTIMIZER_CONFIG.num_steps} steps</span>
                <span>final: {lossData[lossData.length - 1].loss.toFixed(3)}</span>
              </div>
              <svg width={lossW} height={lossH} className="border border-white/10 bg-black w-full">
                <path d={lossPath} fill="none" stroke="white" strokeWidth="1" />
              </svg>
            </div>
            <div className="grid grid-cols-2 gap-1 text-[8px] mt-1">
              <div>
                <p className="text-zinc-500 font-bold">Loop</p>
                <p className="text-zinc-600">1. Forward</p>
                <p className="text-zinc-600">2. -log(P) loss</p>
                <p className="text-zinc-600">3. Backward</p>
                <p className="text-zinc-600">4. Adam step</p>
                <p className="text-zinc-600">5. p.grad = 0</p>
              </div>
              <div>
                <p className="text-zinc-500 font-bold">Adam</p>
                <p className="text-zinc-600">LR: {OPTIMIZER_CONFIG.learning_rate}</p>
                <p className="text-zinc-600">β1: {OPTIMIZER_CONFIG.beta1}</p>
                <p className="text-zinc-600">β2: {OPTIMIZER_CONFIG.beta2}</p>
                <p className="text-zinc-600">ε: {OPTIMIZER_CONFIG.eps}</p>
              </div>
            </div>
          </SideSection>

          {/* Inference */}
          <SideSection title="Inference">
            <div className="text-[8px] space-y-0.5">
              <p className="text-zinc-500">Start with BOS token.</p>
              <p className="text-zinc-500">At each step: logits → softmax(logits/T) → sample.</p>
              <p className="text-zinc-500">Stop when model outputs BOS or max {MODEL_CONFIG.block_size} chars.</p>
              <p className="text-zinc-500 mt-1">Default temperature: 0.6</p>
              <p className="text-zinc-600">Low T → safe. High T → creative.</p>
            </div>
          </SideSection>
        </div>

        {/* ── Main: 3 Live Panels ── */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="px-4 py-2.5 border-b border-white/10 flex items-center gap-3">
            <span className="text-[10px] text-zinc-300 uppercase tracking-widest font-bold">
              Live Pipeline
            </span>
            <span className="text-[9px] text-zinc-600">
              Type a name below: watch it flow through the transformer
            </span>
          </div>

          <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-3 p-3 min-h-0">
            <div className="border border-white/20 overflow-auto bg-black p-5">
              <TokenizerViz />
            </div>
            <div className="border border-white/20 overflow-auto bg-black p-5">
              <EmbeddingViz />
            </div>
            <div className="border border-white/20 overflow-auto bg-black p-5">
              <AttentionViz />
            </div>
          </div>
        </div>
      </div>
    </VisualizerProvider>
  );
}
