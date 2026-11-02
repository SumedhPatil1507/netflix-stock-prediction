"use client";

import Hls from "hls.js";
import { Activity, Radio, ShieldCheck, Signal, Video } from "lucide-react";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const accepted = ".mp4,.mkv,.mov,.json,.srt,.vtt";
type ChatMessage = { role: "user" | "assistant"; content: string };
type Recommendation = { verdict: string; reasoning: string; recommendation: string; next_steps: string[] };

export default function CatalogUpload() {
  const input = useRef<HTMLInputElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const hls = useRef<Hls | null>(null);
  const feedAbort = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState(false);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [searchResult, setSearchResult] = useState<Recommendation | null>(null);
  const [searching, setSearching] = useState(false);
  const [manifest, setManifest] = useState(process.env.NEXT_PUBLIC_DEMO_HLS_URL ?? "");
  const [quality, setQuality] = useState("Auto");
  const [qualityLevels, setQualityLevels] = useState<number[]>([]);
  const [playback, setPlayback] = useState({ current: 0, buffered: 0, bitrate: 0, activeHeight: 0, state: "Idle" });
  const [live, setLive] = useState(false);
  const [liveState, setLiveState] = useState("Disconnected");
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);

  useEffect(() => {
    const element = video.current;
    if (!element || !manifest) return;
    if (Hls.isSupported()) {
      const player = new Hls({ enableWorker: true, lowLatencyMode: true,
        xhrSetup: (xhr) => { if (token.trim()) xhr.setRequestHeader("Authorization", `Bearer ${token.trim()}`); },
      });
      hls.current = player;
      player.loadSource(manifest);
      player.attachMedia(element);
      player.on(Hls.Events.MANIFEST_PARSED, (_event, data) => {
        setQualityLevels(data.levels.map((level) => level.height));
      });
      player.on(Hls.Events.LEVEL_SWITCHED, (_event, data) => {
        const level = player.levels[data.level];
        if (!level) return;
        const bitrate = level.bitrate / 1000;
        setPlayback((state) => ({ ...state, bitrate, activeHeight: level.height }));
        const tokenValue = token.trim();
        if (tokenValue) {
          const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
          const rendition = [360, 720, 1080, 2160].includes(level.height) ? `${level.height}p` : "other";
          void fetch(`${base}/api/v1/telemetry/stream`, {
            method: "POST", headers: { Authorization: `Bearer ${tokenValue}`, "Content-Type": "application/json" },
            body: JSON.stringify({ rendition, bitrate_kbps: bitrate }),
          }).catch(() => undefined);
        }
      });
      player.on(Hls.Events.ERROR, (_event, data) => {
        if (data.fatal) setPlayback((state) => ({ ...state, state: "Stream error" }));
      });
      return () => { player.destroy(); hls.current = null; };
    }
    if (element.canPlayType("application/vnd.apple.mpegurl")) element.src = manifest;
  }, [manifest, token]);

  function changeQuality(value: string) {
    setQuality(value);
    if (!hls.current) return;
    if (value === "Auto") { hls.current.currentLevel = -1; return; }
    const target = value === "4K" ? 2160 : value === "1080p" ? 1080 : 720;
    const levels = hls.current.levels;
    let candidate = -1;
    levels.forEach((level, index) => { if (level.height <= target && (candidate < 0 || level.height > levels[candidate].height)) candidate = index; });
    if (candidate < 0 && levels.length) candidate = levels.reduce((best, level, index, all) => level.height < all[best].height ? index : best, 0);
    hls.current.currentLevel = candidate;
  }

  function updatePlayback() {
    const element = video.current;
    if (!element) return;
    const end = element.buffered.length ? element.buffered.end(element.buffered.length - 1) : 0;
    setPlayback((state) => ({ ...state, current: element.currentTime, buffered: end, state: element.paused ? "Paused" : "Playing" }));
  }

  async function toggleLiveFeed() {
    if (feedAbort.current) {
      feedAbort.current.abort();
      feedAbort.current = null;
      setLive(false);
      setLiveState("Disconnected");
      return;
    }
    if (!token.trim()) { setLiveState("Add a bearer token to connect"); return; }
    const controller = new AbortController();
    feedAbort.current = controller;
    setLive(true);
    setLiveState("Connecting…");
    try {
      const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
      const response = await fetch(`${base}/api/v1/recommendations/stream`, {
        headers: { Authorization: `Bearer ${token.trim()}`, Accept: "text/event-stream" }, signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error(`Live feed failed (${response.status})`);
      setLiveState("Live · waiting for updates");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const line = block.split("\n").find((item) => item.startsWith("data: "));
          if (!line) continue;
          const next = JSON.parse(line.slice(6)) as Recommendation;
          if (next.verdict) { setRecommendation(next); setLiveState("Live · recommendation refreshed"); }
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) setLiveState(error instanceof Error ? error.message : "Live feed disconnected");
    } finally {
      if (feedAbort.current === controller) { feedAbort.current = null; setLive(false); }
    }
  }

  const select = (candidate?: File) => {
    if (!candidate) return;
    setMessage("");
    setFile(candidate);
  };
  const onChange = (event: ChangeEvent<HTMLInputElement>) => select(event.target.files?.[0]);
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    select(event.dataTransfer.files[0]);
  };

  async function upload() {
    if (!file || !token.trim()) {
      setMessage("Choose an asset and enter an admin access token.");
      return;
    }
    setBusy(true);
    setMessage("Uploading and processing asset…");
    const body = new FormData();
    body.append("file", file);
    try {
      const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
      const response = await fetch(`${base}/api/v1/catalog/upload`, {
        method: "POST", headers: { Authorization: `Bearer ${token}` }, body,
      });
      const accepted = await response.json();
      if (!response.ok) throw new Error(accepted.detail ?? `Upload failed (${response.status})`);
      setFile(null);
      if (input.current) input.current.value = "";
      let result = accepted;
      for (let attempt = 0; attempt < 3600 && result.status === "processing"; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        const status = await fetch(`${base}/api/v1/catalog/assets/${accepted.asset_id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        result = await status.json();
        if (!status.ok) throw new Error(result.detail ?? `Could not read processing status (${status.status})`);
        setMessage(result.status === "processing" ? `Processing asset · ${accepted.asset_id}` : "Finalizing…");
      }
      if (result.status === "failed") throw new Error(result.detail ?? "Asset processing failed");
      if (result.status === "processing") throw new Error("Processing is taking longer than expected. Keep the asset ID to check its status later.");
      if (result.manifest) setManifest(`${base}/api/v1/catalog/assets/${result.asset_id}/hls/${result.manifest}`);
      setMessage(`Asset ready · ${result.asset_id}${result.manifest ? ` · HLS ${result.manifest}` : result.indexed ? " · indexed in Qdrant" : " · text parsed"}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed. Check the API connection.");
    } finally {
      setBusy(false);
    }
  }

  async function searchCatalog() {
    if (!question.trim() || !token.trim()) return;
    const current = question.trim();
    setQuestion("");
    setSearching(true);
    try {
      const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
      const response = await fetch(`${base}/api/v1/search/recommendations`, {
        method: "POST", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ query: current, history: chat.slice(-12) }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? `Search failed (${response.status})`);
      const envelope = data as Recommendation;
      setChat((previous) => [...previous, { role: "user" as const, content: current }, { role: "assistant" as const, content: `${envelope.verdict} — ${envelope.reasoning}` }].slice(-12));
      setSearchResult(envelope);
    } catch (error) {
      setChat((previous) => [...previous, { role: "assistant" as const, content: error instanceof Error ? error.message : "Catalog search failed." }].slice(-12));
    } finally {
      setSearching(false);
    }
  }

  return <main className="shell">
    <header><div className="logo">N<span>•</span></div><span className="eyebrow">CONTENT OPERATIONS</span></header>
    <nav className="cockpit-nav"><span className="eyebrow">STREAMING INTELLIGENCE / PRODUCTION COCKPIT</span><a href="/legacy">Legacy UI ↗</a></nav>
    <section className="intro cockpit-intro">
      <p className="eyebrow">LIVE CATALOG / PERSONALIZED PLAYBACK</p>
      <h1>Stories that find<br /><em>their audience.</em></h1>
      <p className="subhead">Adaptive playback, explainable recommendations, and stream health in one live cockpit.</p>
    </section>
    <section className="player-grid" aria-label="Adaptive video player and live telemetry">
      <Card className="player-card">
        <div className="player-top"><span><Video size={15} /> ADAPTIVE PLAYER</span><span className="live-pill"><i /> HLS READY</span></div>
        <video ref={video} controls playsInline onTimeUpdate={updatePlayback} onPlay={updatePlayback} onPause={updatePlayback} />
        <div className="manifest-row"><input value={manifest} onChange={(event) => setManifest(event.target.value)} placeholder="Paste an HLS master manifest URL (.m3u8)" aria-label="HLS manifest URL" /><select value={quality} onChange={(event) => changeQuality(event.target.value)} aria-label="Playback quality"><option>Auto</option><option disabled={!qualityLevels.includes(2160)}>4K</option><option disabled={!qualityLevels.includes(1080)}>1080p</option><option disabled={!qualityLevels.includes(720)}>720p</option></select></div>
        <div className="telemetry-grid"><div><span>QUALITY</span><strong>{quality === "Auto" ? (playback.activeHeight ? `${playback.activeHeight}p auto` : "Auto") : quality}</strong></div><div><span>BITRATE</span><strong>{playback.bitrate ? `${playback.bitrate.toFixed(0)} kbps` : "—"}</strong></div><div><span>PLAYBACK</span><strong>{Math.floor(playback.current / 60)}:{String(Math.floor(playback.current % 60)).padStart(2, "0")}</strong></div><div><span>BUFFERED</span><strong>{Math.max(0, playback.buffered - playback.current).toFixed(1)} sec</strong></div></div>
        {qualityLevels.length > 0 && <p className="quality-list">Available renditions: {qualityLevels.map((level) => `${level}p`).join(" · ")}</p>}
      </Card>
      <Card className="live-card">
        <div className="player-top"><span><Radio size={15} /> RECOMMENDATION FEED</span><Button variant="outline" size="sm" className={`feed-toggle ${live ? "active" : ""}`} onClick={() => void toggleLiveFeed()}>{live ? "Stop feed" : "Connect SSE"}</Button></div>
        <p className="feed-state"><i className={live ? "online" : ""} /> {liveState}</p>
        {recommendation ? <div className="explanation"><span className="eyebrow">TOP MATCH</span><h2>{recommendation.verdict}</h2><p>{recommendation.reasoning}</p><p className="strategy">{recommendation.recommendation}</p><div className="similarity"><Signal size={15} /> {recommendation.reasoning.match(/\((\d+(?:\.\d+)?)%\)/)?.[1] ? `${recommendation.reasoning.match(/\((\d+(?:\.\d+)?)%\)/)?.[1]}%` : "Similarity unavailable"} VECTOR MATCH</div></div> : <div className="empty-feed"><Activity size={25} /><p>Connect the live feed to receive personalized rows as interaction events arrive.</p></div>}
        <div className="privacy-note"><ShieldCheck size={14} /> Authenticated stream · Redis profile updates · Kafka events</div>
      </Card>
    </section>
    <section className="panel" aria-label="Catalog asset upload">
      <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
        <div className="upload-icon">↑</div>
        <strong>{file ? file.name : "Drop your file here"}</strong>
        <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP4, MKV, MOV, JSON, SRT or VTT"}</span>
        <button className="secondary" onClick={() => input.current?.click()} type="button">Browse files</button>
        <input ref={input} type="file" accept={accepted} onChange={onChange} hidden />
      </div>
      <label className="token-label" htmlFor="token">API bearer token <span>(admin required for uploads)</span></label>
      <input id="token" className="token" type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} placeholder="Paste a short-lived bearer token" />
      <button className="primary" type="button" disabled={busy || !file} onClick={upload}>{busy ? "Processing…" : "Start ingestion"}<span>↗</span></button>
      <p className="status" role="status">{message || "Text sidecars are parsed and indexed when Qdrant is configured."}</p>
    </section>
    <section className="search-panel" aria-label="Conversational catalog search">
      <p className="eyebrow">SEARCH & RECOMMEND</p>
      <h2>Tell us what you’re in the mood for.</h2>
      <p className="subhead">Describe a story, genre, runtime, or atmosphere. Suggestions are grounded in the indexed catalog.</p>
      {chat.length > 0 && <div className="conversation" aria-live="polite">{chat.map((item, index) =>
        <div className={`message ${item.role}`} key={`${index}-${item.role}`}><span>{item.role === "user" ? "YOU" : "CATALOG COPILOT"}</span><p>{item.content}</p></div>)}</div>}
      <form className="search-form" onSubmit={(event) => { event.preventDefault(); void searchCatalog(); }}>
        <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Dark sci-fi under 90 minutes with a fast pace…" aria-label="Describe what you want to watch" />
        <button className="primary" type="submit" disabled={searching || !question.trim() || !token.trim()}>{searching ? "Searching…" : "Find titles"}<span>↗</span></button>
      </form>
      {searchResult && <article className="title-card recommendation-card"><span className="eyebrow">EXPLAINABLE MATCH</span><strong>{searchResult.verdict}</strong><p>{searchResult.reasoning}</p><p>{searchResult.recommendation}</p><ol>{searchResult.next_steps.map((step) => <li key={step}>{step}</li>)}</ol></article>}
      {!token.trim() && <p className="status">Enter your admin access token above to search the catalog.</p>}
    </section>
    <footer><span>NETFLIX STREAMING COCKPIT</span><span>HLS · VECTOR RECS · SSE · OPEN TELEMETRY</span></footer>
  </main>;
}
