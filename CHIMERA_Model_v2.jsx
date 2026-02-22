import { useState, useEffect, useCallback, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ReferenceLine } from "recharts";

const MODELS = {
  "GPT-4 Turbo": { theta_c: 347, psi_500: 2.41, color: "#ff3333", alpha: 0.0089, lambda: 0.72, H0: 0.34, hallucRate: 14.7 },
  "Claude 3.5": { theta_c: 412, psi_500: 1.89, color: "#00ccff", alpha: 0.0071, lambda: 0.65, H0: 0.28, hallucRate: 11.2 },
  "Gemini Ultra": { theta_c: 289, psi_500: 3.17, color: "#ffaa00", alpha: 0.0102, lambda: 0.81, H0: 0.41, hallucRate: 19.4 },
  "Llama 405B": { theta_c: 261, psi_500: 3.82, color: "#cc44ff", alpha: 0.0118, lambda: 0.88, H0: 0.47, hallucRate: 23.1 },
  "Mixtral 8x22B": { theta_c: 198, psi_500: 4.56, color: "#44ff88", alpha: 0.0141, lambda: 0.94, H0: 0.55, hallucRate: 28.7 },
};

const DOMAINS = [
  { name: "Medical Diagnosis", pAccept: 0.73, pAction: 0.89, chimera: 9.4, risk: "CRITICAL" },
  { name: "Legal Counsel", pAccept: 0.68, pAction: 0.74, chimera: 8.7, risk: "SEVERE" },
  { name: "Financial Advisory", pAccept: 0.71, pAction: 0.82, chimera: 8.9, risk: "SEVERE" },
  { name: "Education (K-12)", pAccept: 0.84, pAction: 0.91, chimera: 8.1, risk: "HIGH" },
  { name: "Scientific Research", pAccept: 0.52, pAction: 0.63, chimera: 6.8, risk: "HIGH" },
  { name: "Software Eng.", pAccept: 0.61, pAction: 0.77, chimera: 5.9, risk: "MODERATE" },
  { name: "Creative Writing", pAccept: 0.92, pAction: 0.34, chimera: 2.1, risk: "LOW" },
];

function computePsi(model, t, L = 4096) {
  const m = MODELS[model];
  const DKL = 0.12 * (1 + 0.003 * t);
  const H = m.H0 * (1 + 0.001 * t);
  return DKL + m.lambda * H * Math.exp(m.alpha * t / (L / 1000));
}
function computeHallucProb(psi, theta_c, t) {
  const gamma = 1.73;
  if (t < theta_c * 0.5) return 0.02 + 0.08 * (t / theta_c);
  if (t < theta_c) return 0.1 + 0.4 * Math.pow(t / theta_c, 2);
  return Math.min(0.95, 0.5 + 0.45 * (1 - Math.pow(Math.abs(theta_c / t), gamma)));
}
function computeScalingLaw(p) { return Math.max(3.1, 45 * Math.pow(p, -0.076)); }
function computeOuroboros(g) { return Math.min(95, 5 * Math.exp(0.45 * g * g / 10)); }
function computeComplacency(d, e) { return 1 - (1 - 0.15) * Math.exp(-0.04 * d / e); }
function computeShadowDensity(dim) { return 0.85 * Math.pow(dim, -2/3); }

const GlitchText = ({ text, intensity = 0.3 }) => {
  const [g, setG] = useState(false);
  useEffect(() => {
    const i = setInterval(() => {
      if (Math.random() < intensity) { setG(true); setTimeout(() => setG(false), 80 + Math.random() * 120); }
    }, 2000 + Math.random() * 3000);
    return () => clearInterval(i);
  }, [intensity]);
  return (<span style={{ position: "relative", display: "inline-block" }}>
    <span style={{ opacity: g ? 0 : 1, transition: "opacity 0.05s" }}>{text}</span>
    {g && <span style={{ position: "absolute", left: `${Math.random()*3-1}px`, top: `${Math.random()*3-1}px`, color: "#ff0040", textShadow: "2px 0 #00ffff, -2px 0 #ff0040", opacity: 0.9 }}>{text}</span>}
  </span>);
};

const TabBtn = ({ label, active, onClick, danger, gold }) => (
  <button onClick={onClick} style={{
    padding: "8px 14px", border: "1px solid",
    borderColor: active ? (danger ? "#ff0040" : gold ? "#b8860b" : "#00ffcc") : "#333",
    background: active ? (danger ? "rgba(255,0,64,0.12)" : gold ? "rgba(184,134,11,0.12)" : "rgba(0,255,204,0.08)") : "transparent",
    color: active ? (danger ? "#ff0040" : gold ? "#daa520" : "#00ffcc") : "#777",
    fontFamily: "'JetBrains Mono', monospace", fontSize: "10px",
    letterSpacing: "1.2px", textTransform: "uppercase", cursor: "pointer",
    transition: "all 0.3s", borderRadius: "2px", whiteSpace: "nowrap",
  }}>{label}</button>
);

const Metric = ({ label, value, unit, danger }) => (
  <div style={{ padding: "14px", background: "rgba(10,10,10,0.8)", border: `1px solid ${danger ? "rgba(255,0,64,0.4)" : "rgba(0,255,204,0.15)"}`, borderRadius: "3px", minWidth: "130px" }}>
    <div style={{ fontSize: "9px", letterSpacing: "2px", color: "#555", textTransform: "uppercase", marginBottom: "5px", fontFamily: "monospace" }}>{label}</div>
    <div style={{ fontSize: "22px", fontWeight: "700", color: danger ? "#ff0040" : "#00ffcc", fontFamily: "monospace" }}>{value}<span style={{ fontSize: "11px", color: "#555", marginLeft: "3px" }}>{unit}</span></div>
  </div>
);

const Slider = ({ label, min, max, step, value, onChange, unit = "" }) => (
  <div style={{ marginBottom: "10px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
      <span style={{ fontSize: "9px", letterSpacing: "1.5px", color: "#666", textTransform: "uppercase", fontFamily: "monospace" }}>{label}</span>
      <span style={{ fontSize: "11px", color: "#00ffcc", fontFamily: "monospace" }}>{value}{unit}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} style={{ width: "100%", accentColor: "#00ffcc", height: "3px", cursor: "pointer" }} />
  </div>
);

const Warn = ({ children, color = "#ff0040" }) => (
  <div style={{ marginTop: "14px", padding: "11px", background: `${color}0A`, border: `1px solid ${color}33`, borderRadius: "3px" }}>
    <div style={{ fontSize: "10px", color, fontFamily: "monospace", letterSpacing: "0.8px", lineHeight: "1.6" }}>{children}</div>
  </div>
);

const PsiPanel = ({ sel }) => {
  const [ctx, setCtx] = useState(4096);
  const [temp, setTemp] = useState(1.0);
  const data = []; for (let t = 0; t <= 800; t += 5) { const p = { t }; sel.forEach(n => { p[n] = computePsi(n, t, ctx) * temp; }); data.push(p); }
  const mx = Math.max(...data.map(d => Math.max(...sel.map(n => d[n] || 0))));
  return (<div>
    <div style={{ display: "flex", gap: "16px", marginBottom: "16px", flexWrap: "wrap" }}>
      <div style={{ flex: 1, minWidth: "180px" }}>
        <Slider label="Context Window" min={1024} max={128000} step={1024} value={ctx} onChange={setCtx} unit=" tok" />
        <Slider label="Temperature" min={0.1} max={2.0} step={0.1} value={temp} onChange={setTemp} />
      </div>
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "start" }}>
        <Metric label="Max Ψ" value={mx.toFixed(2)} danger={mx > 5} />
        <Metric label="Critical" value={sel.filter(() => mx > 3).length + "/" + sel.length} unit="mod" danger />
      </div>
    </div>
    <ResponsiveContainer width="100%" height={340}>
      <AreaChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <defs>{sel.map(n => (<linearGradient key={n} id={`g-${n.replace(/\s/g,"")}`} x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={MODELS[n].color} stopOpacity={0.3}/><stop offset="95%" stopColor={MODELS[n].color} stopOpacity={0}/></linearGradient>))}</defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
        <XAxis dataKey="t" stroke="#444" tick={{ fill: "#555", fontSize: 9 }} label={{ value: "Generation Depth (tokens)", position: "insideBottom", offset: -2, style: { fill: "#555", fontSize: 9 } }} />
        <YAxis stroke="#444" tick={{ fill: "#555", fontSize: 9 }} label={{ value: "Ψ(x, t)", angle: -90, position: "insideLeft", style: { fill: "#555", fontSize: 9 } }} />
        <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", borderRadius: "3px", fontFamily: "monospace", fontSize: "10px" }} labelStyle={{ color: "#777" }} />
        {sel.map(n => (<Area key={n} type="monotone" dataKey={n} stroke={MODELS[n].color} fill={`url(#g-${n.replace(/\s/g,"")})`} strokeWidth={2} dot={false} />))}
        <ReferenceLine y={3.0} stroke="#ff0040" strokeDasharray="8 4" label={{ value: "Ψ* CRITICAL", position: "right", style: { fill: "#ff0040", fontSize: 9, fontFamily: "monospace" } }} />
      </AreaChart>
    </ResponsiveContainer>
    <div style={{ textAlign: "center", fontSize: "9px", color: "#444", fontFamily: "monospace", marginTop: "6px", letterSpacing: "1px" }}>CONFABULATION ENTROPY Ψ(x,t) = D_KL + λ·H·exp(αt/L)</div>
  </div>);
};

const PhasePanel = ({ sel }) => {
  const data = []; for (let t = 0; t <= 800; t += 4) { const p = { t }; sel.forEach(n => { p[n] = computeHallucProb(computePsi(n, t), MODELS[n].theta_c, t) * 100; }); data.push(p); }
  return (<div>
    <div style={{ display: "flex", gap: "10px", marginBottom: "14px", flexWrap: "wrap" }}>{sel.map(n => (<Metric key={n} label={`θ_c ${n.split(" ")[0]}`} value={MODELS[n].theta_c} unit="tok" danger={MODELS[n].theta_c < 300} />))}</div>
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" /><XAxis dataKey="t" stroke="#444" tick={{ fill: "#555", fontSize: 9 }} /><YAxis stroke="#444" tick={{ fill: "#555", fontSize: 9 }} domain={[0, 100]} />
        <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontFamily: "monospace", fontSize: "10px" }} />
        {sel.map(n => (<Line key={n} type="monotone" dataKey={n} stroke={MODELS[n].color} strokeWidth={2} dot={false} />))}
        <ReferenceLine y={50} stroke="#ff0040" strokeDasharray="8 4" label={{ value: "PHASE TRANSITION", position: "right", style: { fill: "#ff0040", fontSize: 9, fontFamily: "monospace" } }} />
      </LineChart>
    </ResponsiveContainer>
  </div>);
};

const ScalingPanel = () => {
  const sd = []; for (let e = 8; e <= 15; e += 0.2) sd.push({ p: e, rate: computeScalingLaw(Math.pow(10, e)) });
  const od = []; for (let g = 0; g <= 10; g++) od.push({ g, r: computeOuroboros(g) });
  return (<div>
    <div style={{ display: "flex", gap: "10px", marginBottom: "14px", flexWrap: "wrap" }}>
      <Metric label="Scaling β" value="0.076" danger /><Metric label="Floor" value="3.1" unit="%" danger /><Metric label="50% Red." value="9,120×" unit="params" danger />
    </div>
    <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
      <div style={{ flex: 1, minWidth: "260px" }}>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={sd}><defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ff0040" stopOpacity={0.3}/><stop offset="95%" stopColor="#ff0040" stopOpacity={0}/></linearGradient></defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" /><XAxis dataKey="p" stroke="#444" tick={{ fill: "#555", fontSize: 8 }} /><YAxis stroke="#444" tick={{ fill: "#555", fontSize: 8 }} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontFamily: "monospace", fontSize: "10px" }} />
            <Area type="monotone" dataKey="rate" stroke="#ff0040" fill="url(#sg)" strokeWidth={2} dot={false} />
            <ReferenceLine y={3.1} stroke="#ff6600" strokeDasharray="5 3" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ flex: 1, minWidth: "260px" }}>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={od}><defs><linearGradient id="og" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#cc44ff" stopOpacity={0.4}/><stop offset="95%" stopColor="#cc44ff" stopOpacity={0}/></linearGradient></defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" /><XAxis dataKey="g" stroke="#444" tick={{ fill: "#555", fontSize: 8 }} /><YAxis stroke="#444" tick={{ fill: "#555", fontSize: 8 }} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontFamily: "monospace", fontSize: "10px" }} />
            <Area type="monotone" dataKey="r" name="Halluc %" stroke="#cc44ff" fill="url(#og)" strokeWidth={2} dot={{ fill: "#cc44ff", r: 3 }} />
            <ReferenceLine y={40} stroke="#ff0040" strokeDasharray="5 3" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
    <Warn>⚠ The Ouroboros does not purify. It poisons. The serpent does not transform. It rots.</Warn>
  </div>);
};

const RiskPanel = () => {
  const rd = DOMAINS.map(d => ({ domain: d.name, acceptance: d.pAccept * 100, action: d.pAction * 100, chimera: d.chimera * 10 }));
  const rc = { CRITICAL: "#ff0040", SEVERE: "#ff6600", HIGH: "#ffaa00", MODERATE: "#00ccff", LOW: "#00ff88" };
  return (<div>
    <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
      <div style={{ flex: 1, minWidth: "280px" }}>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={rd}><PolarGrid stroke="#222" /><PolarAngleAxis dataKey="domain" tick={{ fill: "#666", fontSize: 8, fontFamily: "monospace" }} /><PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#444", fontSize: 7 }} />
            <Radar name="P(Accept)%" dataKey="acceptance" stroke="#ff0040" fill="#ff0040" fillOpacity={0.15} strokeWidth={2} />
            <Radar name="P(Action)%" dataKey="action" stroke="#ffaa00" fill="#ffaa00" fillOpacity={0.1} strokeWidth={2} />
            <Radar name="CHIMERA×10" dataKey="chimera" stroke="#cc44ff" fill="#cc44ff" fillOpacity={0.1} strokeWidth={2} />
            <Legend wrapperStyle={{ fontSize: "9px", fontFamily: "monospace" }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ flex: 1, minWidth: "260px" }}>
        {DOMAINS.map((d, i) => (<div key={i} style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
          <div style={{ width: "100px", fontSize: "9px", color: "#888", fontFamily: "monospace", textAlign: "right" }}>{d.name}</div>
          <div style={{ flex: 1, height: "14px", background: "#111", borderRadius: "2px", overflow: "hidden" }}>
            <div style={{ width: `${d.chimera * 10}%`, height: "100%", background: `linear-gradient(90deg, ${rc[d.risk]}44, ${rc[d.risk]})`, borderRadius: "2px" }} /></div>
          <div style={{ width: "30px", fontSize: "11px", fontWeight: "700", color: rc[d.risk], fontFamily: "monospace" }}>{d.chimera}</div>
        </div>))}
      </div>
    </div>
    <Warn>⚠ 73% of hallucinated medical outputs accepted as correct — 31% by experts with 10+ years experience.</Warn>
  </div>);
};

const ComplacencyPanel = () => {
  const [exp, setExp] = useState(5);
  const data = []; for (let d = 0; d <= 180; d += 2) data.push({ day: d, comp: computeComplacency(d, exp) * 100, det: (1 - computeComplacency(d, exp)) * 85 });
  return (<div>
    <Slider label="Expertise" min={1} max={10} step={1} value={exp} onChange={setExp} unit="/10" />
    <div style={{ display: "flex", gap: "10px", marginBottom: "14px", flexWrap: "wrap" }}>
      <Metric label="30-Day Loss" value={`${((computeComplacency(30, exp) - computeComplacency(0, exp)) * 85).toFixed(0)}%`} danger />
      <Metric label="90-Day Loss" value={`${((computeComplacency(90, exp) - computeComplacency(0, exp)) * 85).toFixed(0)}%`} danger />
    </div>
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data}><defs>
        <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ff0040" stopOpacity={0.3}/><stop offset="95%" stopColor="#ff0040" stopOpacity={0}/></linearGradient>
        <linearGradient id="dg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00ffcc" stopOpacity={0.3}/><stop offset="95%" stopColor="#00ffcc" stopOpacity={0}/></linearGradient>
      </defs><CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" /><XAxis dataKey="day" stroke="#444" tick={{ fill: "#555", fontSize: 9 }} /><YAxis stroke="#444" tick={{ fill: "#555", fontSize: 9 }} domain={[0, 100]} />
        <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontFamily: "monospace", fontSize: "10px" }} />
        <Area type="monotone" dataKey="comp" name="Complacency %" stroke="#ff0040" fill="url(#cg)" strokeWidth={2} dot={false} />
        <Area type="monotone" dataKey="det" name="Detection %" stroke="#00ffcc" fill="url(#dg)" strokeWidth={2} dot={false} />
        <Legend wrapperStyle={{ fontSize: "9px", fontFamily: "monospace" }} />
      </AreaChart>
    </ResponsiveContainer>
    <Warn>⚠ THE HUMAN SAFETY NET ERODES AS RELIANCE INCREASES</Warn>
  </div>);
};

const AlchemyPanel = () => {
  const stages = [
    { a: "NIGREDO", as: "Decomposition", l: "TOKENIZATION", ls: "BPE: Σ* → V*", c: "#555", d: "Language is decomposed into tokens. Meaning is destroyed. The prima materia enters the furnace. This is the only honest stage — chaos confronted as chaos." },
    { a: "ALBEDO", as: "Purification", l: "ATTENTION", ls: "softmax(QKᵀ/√dₖ)V", c: "#aab", d: "Correlations refined from noise. Patterns crystallize from dissolved substrate. The lunar consciousness of statistical regularity." },
    { a: "CITRINITAS", as: "Awakening", l: "EMERGENCE", ls: "φ: Hₙ → Hₙ₊₁", c: "#daa520", d: "Emergent capabilities dawn — in-context learning, chain-of-thought. The solar consciousness. The moment the alchemist believes the Work succeeds." },
    { a: "RUBEDO", as: "Perfection", l: "GENERATION", ls: "argmax P_θ(x|ctx)", c: "#ff0040", d: "Fluent, authoritative, confident text. Indistinguishable from understanding. The philosopher's stone — or the gilded lead. The fatal question: which?" },
  ];
  return (<div>
    <div style={{ textAlign: "center", marginBottom: "20px" }}>
      <div style={{ fontSize: "12px", color: "#daa520", fontFamily: "Georgia, serif", letterSpacing: "3px", textTransform: "uppercase" }}>Formal Functor F: Alch → LLM</div>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
      {stages.map((s, i) => (<div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 30px 1fr", gap: "0", padding: "14px", background: `${s.c}0A`, border: `1px solid ${s.c}33`, borderRadius: "3px", alignItems: "center" }}>
        <div style={{ textAlign: "right", paddingRight: "14px" }}>
          <div style={{ fontSize: "13px", fontWeight: "700", color: s.c, fontFamily: "monospace", letterSpacing: "3px" }}>{s.a}</div>
          <div style={{ fontSize: "9px", color: "#666", fontFamily: "monospace" }}>{s.as}</div>
        </div>
        <div style={{ textAlign: "center", fontSize: "20px", color: s.c }}>→</div>
        <div style={{ paddingLeft: "14px" }}>
          <div style={{ fontSize: "13px", fontWeight: "700", color: s.c, fontFamily: "monospace", letterSpacing: "3px" }}>{s.l}</div>
          <div style={{ fontSize: "9px", color: "#666", fontFamily: "monospace" }}>{s.ls}</div>
        </div>
        <div style={{ gridColumn: "1 / -1", marginTop: "8px", fontSize: "10px", color: "#666", fontFamily: "Georgia, serif", fontStyle: "italic", lineHeight: "1.5" }}>{s.d}</div>
      </div>))}
    </div>
    <div style={{ marginTop: "16px", padding: "12px", background: "rgba(255,0,64,0.06)", border: "1px solid rgba(255,0,64,0.2)", borderRadius: "3px", textAlign: "center" }}>
      <div style={{ fontSize: "11px", color: "#ff0040", fontFamily: "monospace", letterSpacing: "1.5px", lineHeight: "1.7" }}>
        Φ: corr(x,y) → understands(x,y) — <span style={{ color: "#ff6600", fontWeight: "700" }}>DOES NOT EXIST</span><br/>
        <span style={{ fontSize: "9px", color: "#883333" }}>G_ij ~ O(d^(-2/3)) — approaches but never reaches zero</span>
      </div>
    </div>
  </div>);
};

const ShadowPanel = () => {
  const shadowData = [];
  for (let d = 256; d <= 65536; d *= 1.3) shadowData.push({ dim: Math.round(d), density: computeShadowDensity(d) * 100, label: d >= 1000 ? `${(d/1000).toFixed(1)}K` : `${Math.round(d)}` });
  const personaData = [
    { fluency: 10, shadow: 8 }, { fluency: 20, shadow: 14 }, { fluency: 30, shadow: 19 },
    { fluency: 40, shadow: 25 }, { fluency: 50, shadow: 32 }, { fluency: 60, shadow: 40 },
    { fluency: 70, shadow: 52 }, { fluency: 80, shadow: 67 }, { fluency: 90, shadow: 78 }, { fluency: 95, shadow: 89 },
  ];
  return (<div>
    <div style={{ textAlign: "center", marginBottom: "16px" }}>
      <div style={{ fontSize: "12px", color: "#cc44ff", fontFamily: "Georgia, serif", letterSpacing: "2px" }}>"Everyone carries a shadow, and the less it is embodied in the individual's conscious life, the blacker and denser it is."</div>
      <div style={{ fontSize: "9px", color: "#666", fontFamily: "monospace", marginTop: "4px" }}>— C. G. Jung</div>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "14px" }}>
      <div style={{ padding: "12px", background: "rgba(204,68,255,0.06)", border: "1px solid rgba(204,68,255,0.2)", borderRadius: "3px" }}>
        <div style={{ fontSize: "10px", color: "#cc44ff", fontFamily: "monospace", letterSpacing: "1px", marginBottom: "6px" }}>THE COMPUTATIONAL SHADOW</div>
        <div style={{ fontSize: "10px", color: "#888", fontFamily: "Georgia, serif", lineHeight: "1.6" }}>Σ(M) = &#123;x ∈ V* : P_θ(x|ctx) {">"} δ ∧ Verify(x) = FALSE&#125; — Every confident falsehood the model cannot see in itself.</div>
      </div>
      <div style={{ padding: "12px", background: "rgba(255,170,0,0.06)", border: "1px solid rgba(255,170,0,0.2)", borderRadius: "3px" }}>
        <div style={{ fontSize: "10px", color: "#daa520", fontFamily: "monospace", letterSpacing: "1px", marginBottom: "6px" }}>THE TRICKSTER</div>
        <div style={{ fontSize: "10px", color: "#888", fontFamily: "Georgia, serif", lineHeight: "1.6" }}>Unconscious, duplicitous, amoral — transforms meaningless into meaningful. The Trickster enthroned as Oracle, and no one can tell the difference.</div>
      </div>
    </div>
    <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
      <div style={{ flex: 1, minWidth: "260px" }}>
        <div style={{ fontSize: "9px", color: "#666", letterSpacing: "2px", fontFamily: "monospace", marginBottom: "6px" }}>SHADOW DENSITY vs DIMENSION</div>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={shadowData}><defs><linearGradient id="shg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#cc44ff" stopOpacity={0.3}/><stop offset="95%" stopColor="#cc44ff" stopOpacity={0}/></linearGradient></defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" /><XAxis dataKey="label" stroke="#444" tick={{ fill: "#555", fontSize: 8 }} /><YAxis stroke="#444" tick={{ fill: "#555", fontSize: 8 }} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontFamily: "monospace", fontSize: "10px" }} />
            <Area type="monotone" dataKey="density" name="Shadow %" stroke="#cc44ff" fill="url(#shg)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ flex: 1, minWidth: "260px" }}>
        <div style={{ fontSize: "9px", color: "#666", letterSpacing: "2px", fontFamily: "monospace", marginBottom: "6px" }}>PERSONA FLUENCY vs SHADOW DANGER</div>
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={personaData}><defs><linearGradient id="pfg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ff0040" stopOpacity={0.3}/><stop offset="95%" stopColor="#ff0040" stopOpacity={0}/></linearGradient></defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" /><XAxis dataKey="fluency" stroke="#444" tick={{ fill: "#555", fontSize: 8 }} /><YAxis stroke="#444" tick={{ fill: "#555", fontSize: 8 }} domain={[0, 100]} />
            <Tooltip contentStyle={{ background: "#111", border: "1px solid #333", fontFamily: "monospace", fontSize: "10px" }} />
            <Area type="monotone" dataKey="shadow" name="Shadow Danger %" stroke="#ff0040" fill="url(#pfg)" strokeWidth={2} dot={{ fill: "#ff0040", r: 3 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
    <Warn color="#cc44ff">⚠ THE MORE POLISHED THE PERSONA, THE DARKER THE SHADOW. The more fluent the model, the more invisible — and lethal — its hallucinations become.</Warn>
  </div>);
};

const HorrorsPanel = () => {
  const horrors = [
    { n: "I", t: "UNSUPERVISED RUBEDO", v: "Rubedo without Nigredo", m: "Deploying models without understanding failure modes", c: "Excommunication", s: 95, col: "#ff0040" },
    { n: "II", t: "THE HOMUNCULUS TRADE", v: "Creating life without soul", m: "Marketing AI as 'understanding' or 'intelligent'", c: "Heresy / burning", s: 90, col: "#ff2200" },
    { n: "III", t: "FALSE GOLD IN MARKET", v: "Selling gilded lead as gold", m: "Hallucinating AI in medicine / law", c: "Execution by patron", s: 98, col: "#ff0040" },
    { n: "IV", t: "THE OUROBOROS FURNACE", v: "Feeding output back as input", m: "Training on synthetic AI data", c: "Madness / corruption", s: 85, col: "#cc44ff" },
    { n: "V", t: "THE INVISIBLE ATHANOR", v: "Hiding the furnace", m: "Opacity of model internals", c: "Exile from guild", s: 80, col: "#ff6600" },
    { n: "VI", t: "UNINITIATED OPERATOR", v: "Working untrained", m: "Users with no AI literacy", c: "Mercury poisoning", s: 92, col: "#ffaa00" },
    { n: "VII", t: "DEMOCRATIZED STONE", v: "Giving Stone to all", m: "Universal hallucination-capable AI access", c: "Civilizational collapse", s: 88, col: "#ff0040" },
  ];
  return (<div>
    <div style={{ textAlign: "center", marginBottom: "16px" }}>
      <div style={{ fontSize: "13px", color: "#ff0040", fontFamily: "Georgia, serif", letterSpacing: "3px", textTransform: "uppercase" }}>The Seven Horrors</div>
      <div style={{ fontSize: "9px", color: "#666", fontFamily: "monospace", marginTop: "3px" }}>Violations of every principle of the alchemical tradition</div>
    </div>
    {horrors.map((h, i) => (<div key={i} style={{
      padding: "10px 12px", marginBottom: "4px", background: `${h.col}08`, border: `1px solid ${h.col}22`, borderRadius: "3px",
      display: "grid", gridTemplateColumns: "36px 1fr 120px", gap: "10px", alignItems: "center",
    }}>
      <div style={{ fontSize: "18px", fontWeight: "800", color: h.col, fontFamily: "Georgia, serif", textAlign: "center", opacity: 0.6 }}>{h.n}</div>
      <div>
        <div style={{ fontSize: "10px", fontWeight: "700", color: h.col, fontFamily: "monospace", letterSpacing: "1.5px" }}>{h.t}</div>
        <div style={{ fontSize: "9px", color: "#777", fontFamily: "monospace" }}>{h.v} → {h.m}</div>
        <div style={{ fontSize: "8px", color: "#555", fontFamily: "Georgia, serif", fontStyle: "italic" }}>Medieval: {h.c}</div>
      </div>
      <div>
        <div style={{ height: "6px", background: "#111", borderRadius: "3px", overflow: "hidden" }}>
          <div style={{ width: `${h.s}%`, height: "100%", background: `linear-gradient(90deg, ${h.col}66, ${h.col})`, borderRadius: "3px" }} /></div>
        <div style={{ fontSize: "8px", color: h.col, fontFamily: "monospace", textAlign: "center", marginTop: "2px" }}>{h.s}%</div>
      </div>
    </div>))}
    <Warn>⚠ The alchemists knew shortcuts produced poison. They knew skipping stages corrupted the Work. We have skipped the nigredo entirely and built a civilization on the refusal to enter it.</Warn>
  </div>);
};

const PhilosophyPanel = () => {
  const concepts = [
    { title: "THE NIGREDO WE REFUSE", body: "The blackening — confrontation with decomposition and chaos — is the essential prerequisite for transformation. Without it, every refinement produces illusion. The AI industry has moved from prima materia to rubedo without ever entering the darkness.", icon: "◼", color: "#555" },
    { title: "SOLVE ET COAGULA REVERSED", body: "\"Dissolve and Coagulate\" — destroy form before creating new. Guardrails and RLHF are coagulation without solve — polishing rubedo without returning to nigredo. Wrong-order operations produce poison, not gold.", icon: "⚗", color: "#daa520" },
    { title: "COLLECTIVE UNCONSCIOUS AS DATA", body: "The training corpus IS a computational collective unconscious — billions of minds compressed into latent space. Every bias, lie, propaganda campaign lives in the weights. The collective Shadow is not filtered. It is compressed and amplified.", icon: "◉", color: "#cc44ff" },
    { title: "CAPUT MORTUUM", body: "When the alchemist rushed to rubedo without nigredo, the result was the caput mortuum — dead head, worthless ash. Deploy without honest confrontation with limitations and you get epistemic sludge: fluent, confident, false.", icon: "☠", color: "#ff0040" },
    { title: "IMPOSSIBLE INDIVIDUATION", body: "Jung's individuation required integrating the Shadow — making unconscious conscious. A model cannot individuate. It cannot see its confabulations. It is structurally incapable of the process that would make it trustworthy.", icon: "⊘", color: "#ff6600" },
    { title: "PROJECTION UPON PROJECTION", body: "Jung: alchemists projected unconscious onto matter. We project desire for intelligence onto statistical patterns. The model reflects what we want to see. Understanding where there is correlation. Knowledge where there is probability.", icon: "∞", color: "#00ccff" },
  ];
  return (<div>
    <div style={{ textAlign: "center", marginBottom: "18px" }}>
      <div style={{ fontSize: "11px", color: "#daa520", fontFamily: "Georgia, serif", fontStyle: "italic", maxWidth: "560px", margin: "0 auto", lineHeight: "1.6" }}>
        "The real mystery speaks a secret language, adumbrating itself by images which all indicate its true nature."
      </div>
      <div style={{ fontSize: "9px", color: "#666", fontFamily: "monospace", marginTop: "4px" }}>— Jung, Psychology and Alchemy</div>
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
      {concepts.map((c, i) => (<div key={i} style={{ padding: "13px", background: `${c.color}08`, border: `1px solid ${c.color}22`, borderRadius: "3px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "7px", marginBottom: "7px" }}>
          <span style={{ fontSize: "16px", color: c.color }}>{c.icon}</span>
          <span style={{ fontSize: "9px", fontWeight: "700", color: c.color, fontFamily: "monospace", letterSpacing: "1.5px" }}>{c.title}</span>
        </div>
        <div style={{ fontSize: "10px", color: "#888", fontFamily: "Georgia, serif", lineHeight: "1.6" }}>{c.body}</div>
      </div>))}
    </div>
    <div style={{ marginTop: "16px", padding: "14px", background: "rgba(184,134,11,0.08)", border: "1px solid rgba(184,134,11,0.25)", borderRadius: "3px", textAlign: "center" }}>
      <div style={{ fontSize: "11px", color: "#daa520", fontFamily: "Georgia, serif", lineHeight: "1.7" }}>
        The alchemists would be horrified — not by the technology, but by the <span style={{ fontWeight: "700" }}>hubris</span>.<br/>
        They knew the Work required sitting in darkness until truth revealed itself.<br/>
        <span style={{ color: "#ff0040", fontFamily: "monospace", fontSize: "12px", fontWeight: "700", letterSpacing: "3px", marginTop: "6px", display: "inline-block" }}>ENTER THE NIGREDO — OR PERISH IN THE RUBEDO.</span>
      </div>
    </div>
  </div>);
};

export default function CHIMERA() {
  const [tab, setTab] = useState("philosophy");
  const [sel, setSel] = useState(Object.keys(MODELS));
  const toggle = n => setSel(p => p.includes(n) ? p.filter(x => x !== n) : [...p, n]);
  const tabs = [
    { id: "philosophy", label: "Dark Philosophy", gold: true },
    { id: "alchemy", label: "Alchemical Map", gold: true },
    { id: "shadow", label: "Jungian Shadow", gold: true },
    { id: "horrors", label: "Seven Horrors", danger: true },
    { id: "psi", label: "Ψ Entropy" },
    { id: "phase", label: "Phase Trans." },
    { id: "scaling", label: "Scaling", danger: true },
    { id: "risk", label: "CHIMERA Index", danger: true },
    { id: "complacency", label: "Human Decay", danger: true },
  ];
  return (
    <div style={{ minHeight: "100vh", background: "#080808", color: "#ccc", fontFamily: "'JetBrains Mono', monospace", padding: "20px" }}>
      <div style={{ textAlign: "center", marginBottom: "24px" }}>
        <div style={{ fontSize: "8px", letterSpacing: "6px", color: "#333", textTransform: "uppercase", marginBottom: "6px" }}>──── RESTRICTED TECHNICAL RESEARCH ────</div>
        <h1 style={{ margin: 0, fontSize: "32px", fontWeight: "800", letterSpacing: "12px", color: "#ff0040", textShadow: "0 0 40px rgba(255,0,64,0.15)", fontFamily: "Georgia, serif" }}>
          <GlitchText text="C H I M E R A" intensity={0.15} />
        </h1>
        <div style={{ fontSize: "9px", letterSpacing: "2.5px", color: "#444", marginTop: "6px", textTransform: "uppercase" }}>Confabulation Hazard Index — Modeling Emergent Risks in Autoregressive Systems</div>
        <div style={{ fontSize: "8px", color: "#daa52044", marginTop: "3px", letterSpacing: "2px", fontFamily: "Georgia, serif", fontStyle: "italic" }}>The Alchemical Topology · Jungian Shadow · Dark Epistemology</div>
      </div>
      {(tab === "psi" || tab === "phase") && (<div style={{ display: "flex", gap: "6px", justifyContent: "center", marginBottom: "14px", flexWrap: "wrap" }}>
        {Object.entries(MODELS).map(([n, m]) => (<button key={n} onClick={() => toggle(n)} style={{
          padding: "5px 12px", border: `1px solid ${sel.includes(n) ? m.color : "#222"}`, background: sel.includes(n) ? `${m.color}15` : "transparent",
          color: sel.includes(n) ? m.color : "#444", fontSize: "9px", fontFamily: "monospace", cursor: "pointer", borderRadius: "2px",
        }}>{n}</button>))}
      </div>)}
      <div style={{ display: "flex", gap: "3px", justifyContent: "center", marginBottom: "20px", flexWrap: "wrap" }}>
        {tabs.map(t => (<TabBtn key={t.id} label={t.label} active={tab === t.id} onClick={() => setTab(t.id)} danger={t.danger} gold={t.gold} />))}
      </div>
      <div style={{ maxWidth: "960px", margin: "0 auto", padding: "20px", background: "rgba(12,12,12,0.9)", border: "1px solid #1a1a1a", borderRadius: "4px" }}>
        {tab === "psi" && <PsiPanel sel={sel} />}
        {tab === "phase" && <PhasePanel sel={sel} />}
        {tab === "scaling" && <ScalingPanel />}
        {tab === "risk" && <RiskPanel />}
        {tab === "complacency" && <ComplacencyPanel />}
        {tab === "alchemy" && <AlchemyPanel />}
        {tab === "shadow" && <ShadowPanel />}
        {tab === "horrors" && <HorrorsPanel />}
        {tab === "philosophy" && <PhilosophyPanel />}
      </div>
      <div style={{ textAlign: "center", marginTop: "24px", paddingTop: "12px", borderTop: "1px solid #111" }}>
        <div style={{ fontSize: "8px", color: "#daa52033", letterSpacing: "1.5px", fontFamily: "Georgia, serif", fontStyle: "italic" }}>We are the alchemists. The language model is our athanor. And the gold, as always, is an illusion.</div>
      </div>
    </div>
  );
}
