import { useState, useEffect, useRef, useCallback } from "react";

const I = 0.008, B = 0.004, DT = 0.001, TAU_MAX = 0.5;
const SP_RAD = 10 * Math.PI / 180;

function makeRng(seed) {
  let s = (seed || 1) >>> 0;
  return () => { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >>> 0) / 0xffffffff; };
}
function gauss(rng) {
  const u = Math.max(rng(), 1e-15), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const WIND_PARAMS = {
  light:    { sigma: 0.02, tau: 6.0 },
  moderate: { sigma: 0.06, tau: 6.0 },
  severe:   { sigma: 0.12, tau: 4.0 },
};

const COND_META = {
  no_control: { label: "No Control",     color: "#dc2626" },
  pid_only:   { label: "PID Only",       color: "#d97706" },
  pinn_only:  { label: "PINN Only",      color: "#0284c7" },
  pid_pinn:   { label: "PID + PINN  ★", color: "#16a34a" },
};

function rk4(theta, thetaDot, tauM, tauW) {
  const f = (td) => (tauM - B * td - tauW) / I;
  const k1d = f(thetaDot);
  const k2d = f(thetaDot + DT / 2 * k1d);
  const k3d = f(thetaDot + DT / 2 * k2d);
  const k4d = f(thetaDot + DT * k3d);
  return [
    theta + DT * thetaDot,
    thetaDot + DT / 6 * (k1d + 2*k2d + 2*k3d + k4d)
  ];
}

export default function ArmSim() {
  const [condition, setCondition] = useState("pid_pinn");
  const [windLevel, setWindLevel] = useState("moderate");
  const [gustCount, setGustCount] = useState(2);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [simTime, setSimTime] = useState(0);
  const [phase, setPhase] = useState("READY");
  const [armAngle, setArmAngle] = useState(0);
  const [propSpeed, setPropSpeed] = useState(0);
  const [windTorque, setWindTorque] = useState(0);
  const [motorTorque, setMotorTorque] = useState(0);
  const [errorDeg, setErrorDeg] = useState(0);
  const [chartData, setChartData] = useState([]);
  const [gustMarkers, setGustMarkers] = useState([]);

  const simRef = useRef(null);
  const animRef = useRef(null);
  const propAngleRef = useRef(0);
  const lastFrameRef = useRef(null);
  const chartBuf = useRef([]);

  const buildSim = useCallback(() => {
    const rng = makeRng(42 + gustCount * 7);
    const wp = WIND_PARAMS[windLevel];
    const a = Math.exp(-DT / wp.tau);
    const b_coeff = wp.sigma * Math.sqrt(1 - a * a);
    const gustTimes = Array.from({length: gustCount}, (_, g) =>
      3 + g * (10 / Math.max(gustCount, 1))
    );
    return {
      theta: 0, thetaDot: 0, tauM: 0, windState: 0,
      pidInt: 0, pidPrevErr: 0, pinnPrevErr: 0,
      gyroBuffer: new Array(20).fill(0),
      tauBuffer: new Array(20).fill(0),
      delayBuf: new Array(10).fill(0),
      pinnEst: 0, t: 0, rng, a, b_coeff, gustTimes, wp,
    };
  }, [windLevel, gustCount]);

  const stepSim = useCallback((st, cond) => {
    const { rng, a, b_coeff, gustTimes } = st;
    st.windState = a * st.windState + b_coeff * gauss(rng);
    let tauW = st.windState;
    gustTimes.forEach((gt, i) => { if (st.t >= gt) tauW += (i % 2 === 0 ? 0.10 : -0.09); });

    const gyroNoise = 0.003 / Math.sqrt(DT);
    const gyro = st.thetaDot + 0.001 + gyroNoise * gauss(rng);
    st.gyroBuffer.shift(); st.gyroBuffer.push(gyro);
    st.tauBuffer.shift(); st.tauBuffer.push(st.tauM);

    const tDotNow = st.gyroBuffer[19], tDotPrev = st.gyroBuffer[18];
    const raw = st.tauBuffer[19] - B * tDotNow - I * (tDotNow - tDotPrev) / DT;
    st.pinnEst = 0.15 * raw + 0.85 * st.pinnEst;
    st.delayBuf.shift(); st.delayBuf.push(st.pinnEst);
    const pinnOut = st.delayBuf[0];

    const sp = st.t >= 1.0 ? SP_RAD : 0;
    const clip = v => Math.max(-TAU_MAX, Math.min(TAU_MAX, v));
    let tauCmd = 0;

    if (cond === "no_control") { tauCmd = 0; }
    else if (cond === "pid_only") {
      const e = sp - st.theta; st.pidInt += e * DT;
      tauCmd = clip(1.5*e + 0.80*st.pidInt + 0.10*(e - st.pidPrevErr)/DT);
      st.pidPrevErr = e;
    } else if (cond === "pinn_only") {
      const e = sp - st.theta;
      tauCmd = clip(1.5*e + 0.10*(e - st.pinnPrevErr)/DT + pinnOut);
      st.pinnPrevErr = e;
    } else {
      const e = sp - st.theta; st.pidInt += e * DT;
      tauCmd = clip(1.5*e + 0.80*st.pidInt + 0.10*(e - st.pidPrevErr)/DT + pinnOut);
      st.pidPrevErr = e;
    }

    const alpha = 1 - Math.exp(-DT / 0.020);
    st.tauM = alpha * tauCmd + (1 - alpha) * st.tauM;
    const [nt, ntd] = rk4(st.theta, st.thetaDot, st.tauM, tauW);
    st.theta = nt; st.thetaDot = ntd; st.t += DT;
    return { tauW, tauCmd: st.tauM };
  }, []);

  const reset = useCallback(() => {
    setRunning(false);
    cancelAnimationFrame(animRef.current);
    simRef.current = null;
    setSimTime(0); setArmAngle(0); setPropSpeed(0);
    setWindTorque(0); setMotorTorque(0); setErrorDeg(0);
    setChartData([]); setPhase("READY");
    propAngleRef.current = 0;
    setGustMarkers(Array.from({length: gustCount}, (_, g) =>
      3 + g * (10 / Math.max(gustCount, 1))
    ));
  }, [gustCount]);

  useEffect(() => { reset(); }, [condition, windLevel, gustCount, reset]);

  // Fixed tick: steps = elapsed_ms * speed (since DT = 1ms, this gives exact real-time ratio)
  const tick = useCallback((ts) => {
    if (!simRef.current) return;
    if (!lastFrameRef.current) {
      lastFrameRef.current = ts;
      animRef.current = requestAnimationFrame(tick);
      return;
    }
    const elapsed = ts - lastFrameRef.current;
    lastFrameRef.current = ts;
    const stepsToRun = Math.min(Math.floor(elapsed * speed), 500);
    if (stepsToRun > 0) {
      const st = simRef.current;
      let lastTauW = 0, lastTauCmd = 0;
      for (let i = 0; i < stepsToRun; i++) {
        const r = stepSim(st, condition);
        lastTauW = r.tauW; lastTauCmd = r.tauCmd;
      }
      const tDeg = st.theta * 180 / Math.PI;
      const err = (st.t >= 1 ? 10 : 0) - tDeg;
      propAngleRef.current += (Math.abs(lastTauCmd) / TAU_MAX) * 18 * speed;
      setArmAngle(st.theta); setPropSpeed(Math.abs(lastTauCmd) / TAU_MAX);
      setWindTorque(lastTauW); setMotorTorque(lastTauCmd); setErrorDeg(err);
      setSimTime(st.t);
      const active = st.gustTimes.some(gt => st.t >= gt && st.t < gt + 2);
      setPhase(st.t < 1 ? "SETTLING" : st.t < 3 ? "TRACKING" : active ? "⚡ GUST ACTIVE" : "STABLE");
      if (Math.round(st.t * 100) % 2 === 0) {
        chartBuf.current.push({ t: +st.t.toFixed(2), theta: +Math.max(-30,Math.min(30,tDeg)).toFixed(2), wind: +lastTauW.toFixed(4), err: +Math.abs(err).toFixed(3) });
        if (chartBuf.current.length > 300) chartBuf.current.shift();
        setChartData([...chartBuf.current]);
      }
      if (st.t >= 15) { setRunning(false); setPhase("DONE"); return; }
    }
    animRef.current = requestAnimationFrame(tick);
  }, [condition, speed, stepSim]);

  useEffect(() => {
    if (running) {
      if (!simRef.current) { simRef.current = buildSim(); chartBuf.current = []; lastFrameRef.current = null; }
      animRef.current = requestAnimationFrame(tick);
    } else { cancelAnimationFrame(animRef.current); lastFrameRef.current = null; }
    return () => cancelAnimationFrame(animRef.current);
  }, [running, tick, buildSim]);

  const meta = COND_META[condition];
  const ANG = armAngle * 180 / Math.PI;
  const absErr = Math.abs(errorDeg);
  const errColor = absErr < 1 ? "#16a34a" : absErr < 5 ? "#d97706" : "#dc2626";
  const windMag = Math.min(Math.abs(windTorque) / 0.18, 1);
  const windDir = windTorque >= 0 ? 1 : -1;
  const phaseColor = { READY:"#6b7280", SETTLING:"#9ca3af", TRACKING:"#16a34a", DONE:"#7c3aed" }[phase] || (phase.includes("GUST") ? "#dc2626" : "#16a34a");
  const CX = 220, CY = 180, ARM = 130;
  const ang = -armAngle;
  const mx = CX + ARM * Math.cos(ang), my = CY + ARM * Math.sin(ang);
  const cwx = CX - ARM * Math.cos(ang), cwy = CY - ARM * Math.sin(ang);
  const propA = propAngleRef.current * Math.PI / 180;
  const SP_ANG = -(10 * Math.PI / 180);
  const smx = CX + ARM * Math.cos(SP_ANG), smy = CY + ARM * Math.sin(SP_ANG);
  const scwx = CX - ARM * Math.cos(SP_ANG), scwy = CY - ARM * Math.sin(SP_ANG);

  // ── Palette ────────────────────────────────────────────────────────────────
  const bg       = "#faf8f3";
  const cardBg   = "#ffffff";
  const cardAlt  = "#f5f3ee";
  const border   = "#d4cfc6";
  const borderSub= "#ece8e0";
  const txtPri   = "#1f2937";
  const txtSec   = "#4b5563";
  const txtDim   = "#9ca3af";
  const svgBg    = "#f9f7f2";
  const gridLine = "#ece8e0";
  // ───────────────────────────────────────────────────────────────────────────

  return (
    <div style={{ background:bg, minHeight:"100vh", padding:16, fontFamily:"'Courier New', monospace", color:txtPri, boxSizing:"border-box" }}>
      {/* Header */}
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:12, borderBottom:`1px solid ${borderSub}`, paddingBottom:10 }}>
        <div>
          <div style={{ color:meta.color, fontSize:14, fontWeight:700, letterSpacing:"0.18em" }}>◈ 1-DoF PINN DISTURBANCE OBSERVER</div>
          <div style={{ color:txtDim, fontSize:10, marginTop:3 }}>I=0.008 kg·m²  ·  b=0.004 N·m·s/rad  ·  RK4 @ 1kHz  ·  Dryden turbulence</div>
        </div>
        <div style={{ textAlign:"right" }}>
          <div style={{ color:txtPri, fontSize:18, fontWeight:700 }}>{simTime.toFixed(2)}<span style={{ fontSize:10, color:txtDim }}>s / 15.00s</span></div>
          <div style={{ color:phaseColor, fontSize:10, marginTop:2, fontWeight:600 }}>{phase}</div>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ height:4, background:borderSub, borderRadius:2, marginBottom:14, position:"relative" }}>
        {gustMarkers.map((gt,i) => <div key={i} style={{ position:"absolute", left:`${(gt/15)*100}%`, top:-3, width:1, height:10, background:"#dc2626", opacity:0.7 }}/>)}
        <div style={{ height:"100%", background:`linear-gradient(90deg,${borderSub},${meta.color})`, width:`${(simTime/15)*100}%`, borderRadius:2 }}/>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"440px 1fr", gap:14 }}>
        {/* Left column */}
        <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
          {/* Config panel */}
          <div style={{ background:cardBg, border:`1px solid ${border}`, borderRadius:8, padding:14 }}>
            <div style={{ fontSize:11, color:txtSec, fontWeight:700, letterSpacing:"0.12em", marginBottom:10 }}>── EXPERIMENT CONFIGURATION</div>
            <div style={{ marginBottom:10 }}>
              <div style={{ fontSize:10, color:txtSec, fontWeight:600, marginBottom:5 }}>CONTROL ARCHITECTURE</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:4 }}>
                {Object.entries(COND_META).map(([k,v]) => (
                  <button key={k} onClick={() => setCondition(k)} style={{ background:condition===k?`${v.color}18`:cardAlt, border:`1px solid ${condition===k?v.color:border}`, color:condition===k?v.color:txtSec, padding:"7px 8px", borderRadius:4, cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:10, fontWeight:condition===k?700:400, textAlign:"left" }}>{v.label}</button>
                ))}
              </div>
            </div>
            <div style={{ marginBottom:10 }}>
              <div style={{ fontSize:10, color:txtSec, fontWeight:600, marginBottom:5 }}>WIND INTENSITY</div>
              <div style={{ display:"flex", gap:4 }}>
                {["light","moderate","severe"].map(w => (
                  <button key={w} onClick={() => setWindLevel(w)} style={{ flex:1, background:windLevel===w?"#ede9fe":cardAlt, border:`1px solid ${windLevel===w?"#7c3aed":border}`, color:windLevel===w?"#7c3aed":txtSec, padding:"6px 4px", borderRadius:4, cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:10, fontWeight:windLevel===w?700:400, textTransform:"uppercase" }}>{w}</button>
                ))}
              </div>
            </div>
            <div style={{ marginBottom:12 }}>
              <div style={{ fontSize:10, color:txtSec, fontWeight:600, marginBottom:5 }}>NUMBER OF GUSTS: <span style={{ color:"#dc2626", fontWeight:700 }}>{gustCount}</span></div>
              <input type="range" min={0} max={5} value={gustCount} onChange={e => setGustCount(+e.target.value)} style={{ width:"100%", accentColor:"#dc2626" }}/>
              <div style={{ display:"flex", justifyContent:"space-between", fontSize:9, color:txtDim, marginTop:2 }}><span>0</span><span>5 gusts</span></div>
            </div>
            <div style={{ display:"flex", gap:6, alignItems:"center" }}>
              <button onClick={() => setRunning(r => !r)} style={{ background:running?"#fef2f2":"#f0fdf4", color:running?"#dc2626":"#16a34a", border:`1px solid ${running?"#fca5a5":"#86efac"}`, padding:"7px 16px", borderRadius:4, cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:11, fontWeight:700 }}>{running?"⏸ PAUSE":"▶ PLAY"}</button>
              <button onClick={reset} style={{ background:cardAlt, color:txtSec, border:`1px solid ${border}`, padding:"7px 12px", borderRadius:4, cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:11 }}>⟳ RESET</button>
              <div style={{ display:"flex", gap:3, marginLeft:"auto" }}>
                {[1,2,4,8].map(s => <button key={s} onClick={() => setSpeed(s)} style={{ background:speed===s?"#dbeafe":cardAlt, color:speed===s?"#1d4ed8":txtDim, border:`1px solid ${speed===s?"#93c5fd":border}`, padding:"5px 8px", borderRadius:3, cursor:"pointer", fontFamily:"'Courier New', monospace", fontSize:10, fontWeight:speed===s?700:400 }}>{s}×</button>)}
              </div>
            </div>
          </div>

          {/* Arm visualizer */}
          <div style={{ background:cardBg, border:`1px solid ${meta.color}50`, borderRadius:8, overflow:"hidden" }}>
            <div style={{ background:`${meta.color}12`, borderBottom:`1px solid ${meta.color}30`, padding:"7px 12px", display:"flex", justifyContent:"space-between" }}>
              <span style={{ color:meta.color, fontSize:11, fontWeight:700 }}>{meta.label}</span>
              <span style={{ color:errColor, fontSize:11, fontWeight:700 }}>θ={ANG>=0?"+":""}{ANG.toFixed(1)}°  err={absErr.toFixed(2)}°</span>
            </div>
            <svg width="440" height="340" style={{ display:"block" }}>
              <rect width="440" height="340" fill={svgBg}/>
              {Array.from({length:18},(_,i) => <line key={`h${i}`} x1={0} y1={i*20} x2={440} y2={i*20} stroke={gridLine} strokeWidth={0.6}/>)}
              {Array.from({length:23},(_,i) => <line key={`v${i}`} x1={i*20} y1={0} x2={i*20} y2={340} stroke={gridLine} strokeWidth={0.6}/>)}
              <line x1={0} y1={320} x2={440} y2={320} stroke={border} strokeWidth={1}/>
              {windMag > 0.04 && Array.from({length:8},(_,i) => {
                const y = 40 + i*36, len = windMag*80;
                const sx = windDir>0?8:432, ex = windDir>0?sx+len:sx-len;
                return (
                  <g key={i} opacity={0.2+windMag*0.6}>
                    <line x1={sx} y1={y} x2={ex} y2={y} stroke="#7c3aed" strokeWidth={1.5}/>
                    <polygon points={windDir>0?`${ex},${y} ${ex-8},${y-4} ${ex-8},${y+4}`:`${ex},${y} ${ex+8},${y-4} ${ex+8},${y+4}`} fill="#7c3aed"/>
                  </g>
                );
              })}
              {windMag > 0.05 && <text x={windDir>0?12:428} y={16} fill="#7c3aed" fontSize={9} fontFamily="'Courier New',monospace" textAnchor={windDir>0?"start":"end"} fontWeight="600">τ_wind={windTorque.toFixed(3)} N·m</text>}
              <rect x={CX-24} y={0} width={48} height={10} rx={3} fill={border} stroke={border} strokeWidth={1}/>
              <rect x={CX-4} y={10} width={8} height={CY-90} fill={borderSub} stroke={border} strokeWidth={0.5}/>
              <line x1={scwx} y1={scwy} x2={smx} y2={smy} stroke="rgba(0,0,0,0.08)" strokeWidth={4} strokeLinecap="round"/>
              <circle cx={smx} cy={smy} r={6} fill="none" stroke="rgba(0,0,0,0.12)" strokeWidth={1} strokeDasharray="3,3"/>
              <text x={smx+10} y={smy} fill="rgba(0,0,0,0.3)" fontSize={9} fontFamily="'Courier New',monospace" fontWeight="600">SP 10°</text>
              <line x1={cwx+3} y1={cwy+3} x2={mx+3} y2={my+3} stroke="rgba(0,0,0,0.12)" strokeWidth={10} strokeLinecap="round"/>
              <line x1={cwx} y1={cwy} x2={mx} y2={my} stroke={meta.color} strokeWidth={7} strokeLinecap="round" opacity={0.9}/>
              <line x1={cwx} y1={cwy} x2={mx} y2={my} stroke="rgba(255,255,255,0.4)" strokeWidth={2} strokeLinecap="round"/>
              <circle cx={mx} cy={my} r={18} fill={cardBg} stroke={meta.color} strokeWidth={2.5}/>
              <circle cx={mx} cy={my} r={5} fill={meta.color} opacity={0.7}/>
              {[0,Math.PI/2,Math.PI,3*Math.PI/2].map((offset,i) => {
                const ba = propA+offset;
                return <line key={i} x1={mx+14*Math.cos(ba)} y1={my+14*Math.sin(ba)} x2={mx-14*Math.cos(ba)} y2={my-14*Math.sin(ba)} stroke={meta.color} strokeWidth={3} strokeLinecap="round" opacity={0.5+propSpeed*0.4}/>;
              })}
              {propSpeed > 0.2 && <>
                <circle cx={mx} cy={my} r={22+propSpeed*8} fill="none" stroke={meta.color} strokeWidth={0.8} opacity={0.1+propSpeed*0.12}/>
                <circle cx={mx} cy={my} r={28+propSpeed*14} fill="none" stroke={meta.color} strokeWidth={0.5} opacity={0.06+propSpeed*0.07}/>
              </>}
              <rect x={cwx-10} y={cwy-10} width={20} height={20} rx={4} fill={borderSub} stroke={txtDim} strokeWidth={1.5}/>
              <line x1={cwx-6} y1={cwy-6} x2={cwx+6} y2={cwy+6} stroke={txtDim} strokeWidth={1}/>
              <line x1={cwx+6} y1={cwy-6} x2={cwx-6} y2={cwy+6} stroke={txtDim} strokeWidth={1}/>
              <circle cx={CX} cy={CY} r={12} fill={borderSub} stroke={txtSec} strokeWidth={2.5}/>
              <circle cx={CX} cy={CY} r={5} fill={txtSec}/>
              <circle cx={CX} cy={CY} r={2} fill={cardBg}/>
              <text x={12} y={328} fill={meta.color} fontSize={9} fontFamily="'Courier New',monospace" fontWeight="600">τ_motor={motorTorque.toFixed(3)} N·m</text>
              <text x={428} y={328} fill={errColor} fontSize={9} fontFamily="'Courier New',monospace" textAnchor="end" fontWeight="600">|err|={absErr.toFixed(2)}°</text>
              <rect x={4} y={332} width={Math.min(absErr/20*432,432)} height={4} rx={2} fill={errColor} opacity={0.5}/>
            </svg>
          </div>
        </div>

        {/* Right column */}
        <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
          {/* Stat cards */}
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8 }}>
            {[
              {label:"ANGLE", value:`${ANG>=0?"+":""}${ANG.toFixed(2)}°`, color:meta.color},
              {label:"ERROR", value:`${absErr.toFixed(3)}°`, color:errColor},
              {label:"τ MOTOR", value:`${motorTorque.toFixed(3)} N·m`, color:"#d97706"},
              {label:"τ WIND", value:`${windTorque.toFixed(3)} N·m`, color:"#7c3aed"},
            ].map(s => (
              <div key={s.label} style={{ background:cardBg, border:`1px solid ${border}`, borderRadius:6, padding:"10px 12px" }}>
                <div style={{ fontSize:9, color:txtDim, fontWeight:600, letterSpacing:"0.12em", marginBottom:4 }}>{s.label}</div>
                <div style={{ fontSize:15, color:s.color, fontWeight:700 }}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* Angle chart */}
          <div style={{ background:cardBg, border:`1px solid ${border}`, borderRadius:8, padding:"12px 14px", flex:1 }}>
            <div style={{ fontSize:12, color:txtPri, fontWeight:700, letterSpacing:"0.12em", marginBottom:6 }}>── ANGLE TRACKING  θ(t)</div>
            <svg width="100%" height="130" viewBox="0 0 580 130" style={{ display:"block" }}>
              <defs>
                <linearGradient id="thetaFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={meta.color} stopOpacity={0.18}/>
                  <stop offset="100%" stopColor={meta.color} stopOpacity={0}/>
                </linearGradient>
              </defs>
              {[-20,-10,0,10,20].map(v => {
                const y = 65-(v/25)*55;
                return <g key={v}><line x1={0} y1={y} x2={580} y2={y} stroke={gridLine} strokeWidth={0.8}/><text x={2} y={y-2} fill={txtDim} fontSize={9} fontFamily="'Courier New',monospace">{v}°</text></g>;
              })}
              <line x1={0} y1={65-(10/25)*55} x2={580} y2={65-(10/25)*55} stroke="rgba(0,0,0,0.15)" strokeWidth={1} strokeDasharray="4,4"/>
              {chartData.slice(-80).length > 2 && (() => {
                const d = chartData.slice(-80);
                const path = d.map((pt,i) => { const x=(i/(d.length-1))*580, y=65-(pt.theta/25)*55; return `${i===0?"M":"L"} ${x.toFixed(1)} ${y.toFixed(1)}`; }).join(" ");
                return <>
                  <path d={path+` L 580 65 L 0 65 Z`} fill="url(#thetaFill)"/>
                  <path d={path} fill="none" stroke={meta.color} strokeWidth={2}/>
                </>;
              })()}
            </svg>
          </div>

          {/* Wind chart */}
          <div style={{ background:cardBg, border:`1px solid ${border}`, borderRadius:8, padding:"12px 14px" }}>
            <div style={{ fontSize:12, color:txtPri, fontWeight:700, letterSpacing:"0.12em", marginBottom:6 }}>── WIND DISTURBANCE  τ_wind(t)  [N·m]</div>
            <svg width="100%" height="80" viewBox="0 0 580 80" style={{ display:"block" }}>
              {[-0.2,-0.1,0,0.1,0.2].map(v => {
                const y=40-(v/0.25)*35;
                return <g key={v}><line x1={0} y1={y} x2={580} y2={y} stroke={gridLine} strokeWidth={0.8}/><text x={2} y={y-2} fill={txtDim} fontSize={9} fontFamily="'Courier New',monospace">{v.toFixed(1)}</text></g>;
              })}
              {chartData.slice(-80).length > 2 && (() => {
                const d = chartData.slice(-80);
                const path = d.map((pt,i) => { const x=(i/(d.length-1))*580, y=40-(pt.wind/0.25)*35; return `${i===0?"M":"L"} ${x.toFixed(1)} ${y.toFixed(1)}`; }).join(" ");
                return <path d={path} fill="none" stroke="#7c3aed" strokeWidth={1.8}/>;
              })()}
              {gustMarkers.map((gt,i) => {
                const d = chartData.slice(-80);
                if (!d.length) return null;
                const tMin=d[0].t, tMax=d[d.length-1].t;
                if (gt<tMin||gt>tMax) return null;
                const x=((gt-tMin)/(tMax-tMin))*580;
                return <line key={i} x1={x} y1={0} x2={x} y2={80} stroke="#dc2626" strokeWidth={1} strokeDasharray="3,3" opacity={0.6}/>;
              })}
            </svg>
          </div>

          {/* Error chart */}
          <div style={{ background:cardBg, border:`1px solid ${border}`, borderRadius:8, padding:"12px 14px" }}>
            <div style={{ fontSize:12, color:txtPri, fontWeight:700, letterSpacing:"0.12em", marginBottom:6 }}>── TRACKING ERROR  |θ_ref − θ|  [°]</div>
            <svg width="100%" height="70" viewBox="0 0 580 70" style={{ display:"block" }}>
              {[0,5,10].map(v => {
                const y=65-(v/12)*60;
                return <g key={v}><line x1={0} y1={y} x2={580} y2={y} stroke={gridLine} strokeWidth={0.8}/><text x={2} y={y-2} fill={txtDim} fontSize={9} fontFamily="'Courier New',monospace">{v}°</text></g>;
              })}
              {chartData.slice(-150).length > 2 && (() => {
                const d = chartData.slice(-150);
                const path = d.map((pt,i) => { const x=(i/(d.length-1))*580, y=65-(Math.min(pt.err,12)/12)*60; return `${i===0?"M":"L"} ${x.toFixed(1)} ${y.toFixed(1)}`; }).join(" ");
                return <path d={path} fill="none" stroke={errColor} strokeWidth={1.5}/>;
              })()}
            </svg>
          </div>

          {/* Legend */}
          <div style={{ display:"flex", gap:16, justifyContent:"center", flexWrap:"wrap", padding:"6px 0", borderTop:`1px solid ${borderSub}`, fontSize:10, color:txtSec, fontWeight:500 }}>
            <span>▏ Red markers = gust events</span>
            <span>▏ Ghost arm = setpoint 10°</span>
            <span>▏ Purple arrows = wind direction &amp; magnitude</span>
            <span>▏ Prop blur speed ∝ motor torque</span>
          </div>
        </div>
      </div>
    </div>
  );
}
