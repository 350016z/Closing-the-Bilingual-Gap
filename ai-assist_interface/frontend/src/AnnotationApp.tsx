'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Moon, Sun, Trash2, Plus, Minus, AlertTriangle, History,
  ChevronLeft, ChevronRight, Save, LogOut, Info,
  CheckCircle2, AlertCircle, Clock, Languages, RotateCcw
} from 'lucide-react';

/**
 * 工具函數：處理索引轉換 (UTF-16 <-> CodePoint)
 */
const toUTF16Index = (str: string, cpIndex: number) => {
  if (!str) return 0;
  return Array.from(str).slice(0, cpIndex).join('').length;
};

const toCodePointIndex = (str: string, utf16Index: number) => {
  if (!str) return 0;
  return Array.from(str.substring(0, utf16Index)).length;
};

const SEVERITIES = [
  { label: 'Minor', value: 'Minor', color: 'bg-yellow-200 text-yellow-800 dark:bg-yellow-500/30 dark:text-yellow-200' },
  { label: 'Major', value: 'Major', color: 'bg-orange-200 text-orange-800 dark:bg-orange-500/30 dark:text-orange-200' }
];

interface Span {
  id: string;
  start: number;
  end: number;
  text: string;
  severity: string;
}

interface Log {
  id: string;
  timestamp: string;
  action: string;
  details: string;
  spanId?: string;
  metadata?: { totalDelta?: number; originalText?: string; newText?: string };
}

interface DataRow {
  id: string;
  mt: string;
  src: string;
  errors?: { start: number; end: number; error_text: string; severity: string }[];
  score7?: number;
  explanation?: string;
  after_errors?: { start: number; end: number; text: string; severity: string, id: string }[];
  after_score7?: number;
  after_explanation?: string;
}

export default function AnnotationApp() {
  const [isDark, setIsDark] = useState(true);
  const [showMain, setShowMain] = useState(false);
  const [annotator, setAnnotator] = useState('');
  const [file, setFile] = useState('');

  const [dataRows, setDataRows] = useState<DataRow[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [spans, setSpans] = useState<Span[]>([]);
  const [logs, setLogs] = useState<Log[]>([]);
  const [score7, setScore7] = useState(6);
  const [explanation, setExplanation] = useState('');
  const [timeSpent, setTimeSpent] = useState(0);
  const [targetIdxInput, setTargetIdxInput] = useState('');
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);
  const spanRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

  const currentData = dataRows[currentIdx];
  const tgtRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const annotatorOptions = ['demo', 'rater1', 'rater2', 'rater3'];
  const fileOptions = [
    'gemini-3-pro-preview_align_topsen5_0.8_topword5_0.9.jsonl',
    'gpt-5.2_align_topsen5_0.8_topword5_0.9.jsonl',
  ];

  useEffect(() => {
    if (isDark) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [isDark]);

  useEffect(() => {
    if (showMain && currentData) {
      timerRef.current = setInterval(() => setTimeSpent(t => t + 1), 1000);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [showMain, currentData]);

  useEffect(() => {
    if (!file) return;
    fetch(`/annotation_files/${file}`)
      .then(res => res.text())
      .then(text => {
        const rows = text.trim().split('\n').filter(l => l).map(line => JSON.parse(line));
        setDataRows(rows);
        setCurrentIdx(0);
      });
  }, [file]);

  useEffect(() => {
    if (currentData?.id) setTargetIdxInput(currentData.id);
  }, [currentData]);

  useEffect(() => {
    if (selectedSpanId && spanRefs.current[selectedSpanId]) {
      spanRefs.current[selectedSpanId]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [selectedSpanId]);

  useEffect(() => {
    if (!currentData || !file || !annotator) return;
    setSpans([]);
    setLogs([]);
    fetch(`/annotation/load?filename=${file}&rater=${annotator}&id=${currentData.id}`)
      .then(res => res.json())
      .then(res => {
        if (res.record) {
          // Priority: after_errors -> spans -> empty
          const rawSpans = res.record.after_errors || res.record.spans || [];
          const loadedSpans = rawSpans.map((s: any) => ({
            ...s,
            start: toUTF16Index(currentData.mt, s.start),
            end: toUTF16Index(currentData.mt, s.end),
            id: s.id || Math.random().toString(36).substr(2, 9)
          }));
          setSpans(loadedSpans);

          setLogs((res.record.logs || []).map((l: any, i: number) => ({ ...l, id: l.id || `log-${i}` })));

          // Load score: after_score7 -> (fallback from user request: original score if not changed)
          // But actually, if record exists, it means we might have saved it. 
          // If after_score7 is undefined, it might mean it wasn't changed? 
          // User said: "if not changed, it is the original score7".
          // So we check after_score7 first.
          const savedScore = res.record.after_score7 ?? res.record.score7 ?? 0;
          setScore7(savedScore);

          const savedExplanation = res.record.after_explanation ?? res.record.explanation ?? '';
          setExplanation(savedExplanation);

          setTimeSpent(res.record.time_spent ?? res.record.timeSpent ?? 0);
        } else if (currentData) {
          // Pre-fill from currentData if no saved record
          const initSpans = (currentData.errors || []).map((e: any) => ({
            id: Math.random().toString(36).substr(2, 9),
            start: toUTF16Index(currentData.mt, e.start),
            end: toUTF16Index(currentData.mt, e.end),
            text: e.error_text,
            severity: (e.severity === 'Major' || e.severity === 'Minor') ? e.severity : 'Minor',
          }));
          setSpans(initSpans);

          // Pre-fill Score and Explanation from original data
          setScore7(currentData.score7 ?? 0);
          setExplanation(currentData.explanation ?? '');
          setTimeSpent(0);
        }
      });
  }, [currentData, file, annotator]);

  const saveAnnotation = async () => {
    if (!currentData || !file || !annotator) return;
    const diffParams = spans.map(s => ({
      ...s,
      start: toCodePointIndex(currentData.mt, s.start),
      end: toCodePointIndex(currentData.mt, s.end),
    }));
    const record = {
      ...currentData,
      after_errors: diffParams,
      logs: logs.map(({ id: _id, ...rest }) => rest),
      rater: annotator,
      after_score7: score7,
      after_explanation: explanation,
      time_spent: timeSpent,
    };
    await fetch("/annotation/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file, rater: annotator, record: record })
    });
  };

  const resetAnnotation = () => {
    if (!currentData) return;
    if (confirm('Are you sure you want to reset all changes? This cannot be undone.')) {
      setLogs([]);
      setTimeSpent(0);

      // Reset Spans from original errors
      const initSpans = (currentData.errors || []).map((e: any) => ({
        id: Math.random().toString(36).substr(2, 9),
        start: toUTF16Index(currentData.mt, e.start),
        end: toUTF16Index(currentData.mt, e.end),
        text: e.error_text,
        severity: (e.severity === 'Major' || e.severity === 'Minor') ? e.severity : 'Minor',
      }));
      setSpans(initSpans);

      // Reset Score & Explanation
      setScore7(currentData.score7 ?? 0);
      setExplanation(currentData.explanation ?? '');

      addLog('Reset', 'Reset all annotations to original state');
    }
  };

  const addLog = (action: string, details: string, spanId?: string, metadata?: { totalDelta?: number; originalText?: string; newText?: string }) => {
    const NOW = new Date();
    const timestamp = `${NOW.getHours().toString().padStart(2, '0')}:${NOW.getMinutes().toString().padStart(2, '0')}:${NOW.getSeconds().toString().padStart(2, '0')}`;

    setLogs(prev => {
      // Coalesce 'Resize' and 'Move' if on the same span
      if ((action === 'Resize' || action === 'Move') && spanId && prev.length > 0) {
        const lastLog = prev[0];
        if (lastLog.action === action && lastLog.spanId === spanId) {
          // Special handling for Move to accumulate steps
          if (action === 'Move' && metadata?.totalDelta && lastLog.metadata?.totalDelta) {
            const newTotalDelta = lastLog.metadata.totalDelta + metadata.totalDelta;

            // If moved back to original position (delta 0), remove the log
            if (newTotalDelta === 0) {
              return prev.slice(1);
            }

            const directionStr = newTotalDelta > 0 ? 'Right' : 'Left';
            const steps = Math.abs(newTotalDelta);
            const newDetails = `Moved "${lastLog.metadata.originalText}" ${directionStr} ${steps} Steps`;

            return [{
              ...lastLog,
              details: newDetails,
              timestamp,
              metadata: { ...lastLog.metadata, totalDelta: newTotalDelta }
            }, ...prev.slice(1)];
          }

          // Special handling for Resize to accumulate changes from original state
          if (action === 'Resize' && metadata?.newText && lastLog.metadata?.originalText) {
            const origin = lastLog.metadata.originalText;
            const target = metadata.newText;

            // If resized back to original text, remove the log
            if (origin === target) {
              return prev.slice(1);
            }

            const newDetails = `Resized "${origin}" to "${target}"`;
            return [{
              ...lastLog,
              details: newDetails,
              timestamp,
              metadata: { ...lastLog.metadata, newText: target }
            }, ...prev.slice(1)];
          }

          // Update the last log entry instead of adding a new one
          return [{ ...lastLog, details, timestamp }, ...prev.slice(1)];
        }
      }
      return [{ id: Date.now().toString(36), timestamp, action, details, spanId, metadata }, ...prev];
    });
  };

  const handleSelection = () => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !tgtRef.current) return;
    const range = sel.getRangeAt(0);
    if (!tgtRef.current.contains(range.commonAncestorContainer)) return;

    let offset = 0, start = -1, end = -1;
    const walker = document.createTreeWalker(tgtRef.current, NodeFilter.SHOW_TEXT, null);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const nodeLen = node.nodeValue?.length || 0;
      if (node === range.startContainer) start = offset + range.startOffset;
      if (node === range.endContainer) end = offset + range.endOffset;
      offset += nodeLen;
      if (start !== -1 && end !== -1) break;
    }

    if (start !== -1 && end !== -1 && start < end) {
      const text = currentData.mt.substring(start, end);
      const newSpan = { id: Math.random().toString(36).substr(2, 9), start, end, text, severity: 'Minor' };
      setSpans(prev => [...prev, newSpan]);
      setSelectedSpanId(newSpan.id);
      addLog('Add', `Marked "${text}"`);
      sel.removeAllRanges();
    }
  };

  const removeSpan = (id: string) => {
    const s = spans.find(x => x.id === id);
    setSpans(prev => prev.filter(x => x.id !== id));
    if (id === selectedSpanId) setSelectedSpanId(null);
    if (s) addLog('Remove', `Removed "${s.text}"`);
  };

  const updateSeverity = (id: string, sev: string) => {
    setSpans(prev => prev.map(s => s.id === id ? { ...s, severity: sev } : s));
    addLog('Severity', `Changed "${spans.find(x => x.id === id)?.text}" to ${sev}`);
  };

  const resizeSpan = (id: string, side: 'start' | 'end', change: number) => {
    const s = spans.find((x) => x.id === id);
    if (!s) return;

    // Convert to Code Point Index to handle surrogate pairs
    const cpStart = toCodePointIndex(currentData.mt, s.start);
    const cpEnd = toCodePointIndex(currentData.mt, s.end);

    let newCpStart = cpStart;
    let newCpEnd = cpEnd;

    if (side === 'start') newCpStart += change;
    if (side === 'end') newCpEnd += change;

    // Convert back to UTF-16 Index
    const newStart = toUTF16Index(currentData.mt, newCpStart);
    const newEnd = toUTF16Index(currentData.mt, newCpEnd);

    // Validation
    if (newStart >= newEnd || newStart < 0 || newEnd > currentData.mt.length) return;

    const newText = currentData.mt.substring(newStart, newEnd);

    // Log outside
    addLog('Resize', `Resized "${s.text}" to "${newText}"`, id, { originalText: s.text, newText: newText });

    // Update
    setSpans((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, start: newStart, end: newEnd, text: newText } : item
      )
    );
  };

  const moveSpan = (id: string, direction: number) => {
    const s = spans.find((x) => x.id === id);
    if (!s) return;

    // Convert to Code Point Index to handle surrogate pairs
    const cpStart = toCodePointIndex(currentData.mt, s.start);
    const cpEnd = toCodePointIndex(currentData.mt, s.end);
    const totalCodePoints = toCodePointIndex(currentData.mt, currentData.mt.length);

    const newCpStart = cpStart + direction;
    const newCpEnd = cpEnd + direction;

    // Strict Boundary Check in Code Points
    if (newCpStart < 0 || newCpEnd > totalCodePoints) return;

    // Convert back to UTF-16 Index
    const newStart = toUTF16Index(currentData.mt, newCpStart);
    const newEnd = toUTF16Index(currentData.mt, newCpEnd);

    // Validation
    if (newStart < 0 || newEnd > currentData.mt.length) return;

    const newText = currentData.mt.substring(newStart, newEnd);

    // Log Outside
    addLog('Move', `Moved "${s.text}" ${direction === 1 ? 'Right' : 'Left'} 1 Steps`, id, { totalDelta: direction, originalText: s.text });

    // Update
    setSpans((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, start: newStart, end: newEnd, text: newText } : item
      )
    );
  };

  const renderText = () => {
    if (!currentData) return null;
    const text = currentData.mt;
    const charMap = new Array(text.length).fill(null);
    spans.forEach(s => { for (let i = s.start; i < s.end; i++) charMap[i] = s; });

    const elements = [];
    let i = 0;
    while (i < text.length) {
      const s = charMap[i];
      if (!s) { elements.push(<span key={i}>{text[i]}</span>); i++; }
      else {
        let j = i;
        while (j < text.length && charMap[j]?.id === s.id) j++;
        const sevStyle = SEVERITIES.find(v => v.value === s.severity) || SEVERITIES[0];
        if (!sevStyle) { i++; continue; }
        const isSelected = s.id === selectedSpanId;
        elements.push(
          <mark key={i}
            onClick={(e) => { e.stopPropagation(); setSelectedSpanId(s.id); }}
            className={`transition-all px-0.5 rounded cursor-pointer ${sevStyle.color} decoration-current shadow-sm ${isSelected ? 'ring-2 ring-blue-600 z-10 relative' : ''}`}>
            {text.substring(i, j)}
          </mark>
        );
        i = j;
      }
    }
    return elements;
  };

  if (!showMain) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-neutral-950 flex items-center justify-center p-6 transition-colors duration-300">
        <div className="w-full max-w-md bg-white dark:bg-neutral-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-neutral-800 p-8 space-y-6">
          <div className="text-center space-y-2 relative">
            <button onClick={() => setIsDark(!isDark)} className="absolute right-0 top-0 p-2 hover:bg-slate-100 dark:hover:bg-neutral-800 rounded-lg transition-colors">{isDark ? <Sun size={20} className="text-yellow-500" /> : <Moon size={20} className="text-slate-600" />}</button>
            <h1 className="text-2xl font-extrabold text-slate-900 dark:text-white tracking-tighter">SpanEval Lab</h1>
            <p className="text-[12px] text-slate-500 dark:text-neutral-400">AI-Assisted Error Span Annotation for Taiwanese MT Evaluation</p>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-bold mb-2 text-slate-700 dark:text-slate-300">Rater Identity</label>
              <select className="w-full bg-slate-100 dark:bg-neutral-800 border-none rounded-xl p-3 outline-none text-slate-700 dark:text-slate-300 text-xs"
                value={annotator} onChange={e => setAnnotator(e.target.value)}>
                <option value="">Select...</option>
                {annotatorOptions.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-bold mb-2 text-slate-700 dark:text-slate-300">Dataset File</label>
              <select className="w-full bg-slate-100 dark:bg-neutral-800 border-none rounded-xl p-3 outline-none text-slate-700 dark:text-slate-300 text-xs"
                value={file} onChange={e => setFile(e.target.value)}>
                <option value="">Select...</option>
                {fileOptions.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
            <button onClick={() => setShowMain(true)} disabled={!file || !annotator} className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-bold py-4 rounded-xl shadow-lg transition-all">Start Session</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-slate-50 dark:bg-neutral-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans overflow-hidden transition-colors duration-300">
      {/* Header */}
      <header className="h-14 shrink-0 bg-white dark:bg-neutral-900 border-b border-slate-200 dark:border-neutral-800 px-6 flex items-center justify-between z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 bg-blue-600 border border-blue-400 rounded flex items-center justify-center shadow-[0_0_8px_rgba(59,130,246,0.6)] overflow-hidden transition-all">
            <img src="/favicon.ico" alt="Logo" className="w-5 h-5 object-contain brightness-0 invert" />
          </div>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-tight">ErrorSpan Annotator</h2>
            <div className="text-[9px] text-slate-400 font-mono uppercase tracking-widest">{annotator} • {file}</div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3 px-3 py-1 bg-slate-100 dark:bg-neutral-800 rounded-full border border-slate-200 dark:border-neutral-700 text-[11px] font-mono">
            <div className="flex items-center gap-1.5 text-blue-500 font-bold clickable cursor-pointer hover:text-blue-600" onClick={() => setTimeSpent(0)} title="Reset Timer">
              <Clock size={12} />
              <span>{Math.floor(timeSpent / 60)}m {timeSpent % 60}s</span>
              <RotateCcw size={10} className="ml-1 opacity-50 hover:opacity-100" />
            </div>
            <div className="w-px h-3 bg-slate-300 dark:bg-neutral-600"></div>
            <div className="font-bold text-blue-600">{currentIdx + 1} / {dataRows.length}</div>
          </div>
          <button onClick={() => setIsDark(!isDark)} className="p-2 hover:bg-slate-100 dark:hover:bg-neutral-800 rounded-lg transition-colors">{isDark ? <Sun size={16} className="text-yellow-500" /> : <Moon size={16} className="text-slate-600" />}</button>
          <button className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-50 hover:bg-orange-100 dark:bg-orange-950/30 text-orange-600 dark:text-orange-400 rounded-lg text-[11px] font-bold transition-all" onClick={resetAnnotation} title="Reset to Original">
            <RotateCcw size={12} /> Reset
          </button>
          <button className="flex items-center gap-1.5 px-3 py-1.5 bg-red-50 hover:bg-red-100 dark:bg-red-950/30 text-red-600 dark:text-red-400 rounded-lg text-[11px] font-bold transition-all" onClick={() => setShowMain(false)}>
            <LogOut size={12} /> Exit
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left: Content Area */}
        <section className="flex-1 flex flex-col p-4 pt-2 gap-2 overflow-hidden">
          {/* Navigation & Global Actions */}
          <div className="flex justify-between items-center bg-white dark:bg-neutral-900 p-2 px-3 rounded-xl border border-slate-200 dark:border-neutral-800 shadow-sm shrink-0">
            <button className="flex items-center gap-1 px-4 py-1.5 rounded-lg bg-slate-100 dark:bg-neutral-800 hover:bg-slate-200 dark:hover:bg-neutral-700 text-xs font-bold disabled:opacity-30 transition-all"
              disabled={currentIdx === 0} onClick={() => { saveAnnotation(); setCurrentIdx(c => c - 1); }}><ChevronLeft size={14} /> Prev</button>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-400 font-mono uppercase tracking-tighter">Index:</span>
                <input type="number" className="w-12 h-7 bg-slate-50 dark:bg-neutral-800 border border-slate-200 dark:border-neutral-700 rounded text-center text-xs font-mono outline-none focus:ring-1 focus:ring-blue-500"
                  value={targetIdxInput} onChange={e => setTargetIdxInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') {
                      const targetId = targetIdxInput.trim();
                      const idx = dataRows.findIndex(r => r.id === targetId);
                      if (idx !== -1) {
                        saveAnnotation();
                        setCurrentIdx(idx);
                        setTargetIdxInput('');
                      }
                    }
                  }} />
              </div>
              <button onClick={saveAnnotation} className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-1.5 rounded-lg font-bold text-xs flex items-center gap-2 shadow-md active:scale-95 transition-all"><Save size={14} /> Save Current</button>
            </div>
            <button className="flex items-center gap-1 px-4 py-1.5 rounded-lg bg-slate-100 dark:bg-neutral-800 hover:bg-slate-200 dark:hover:bg-neutral-700 text-xs font-bold disabled:opacity-30 transition-all"
              disabled={currentIdx === dataRows.length - 1} onClick={() => { saveAnnotation(); setCurrentIdx(c => c + 1); }}>Next <ChevronRight size={14} /></button>
          </div>

          {/* Texts Area */}
          <div className="flex-1 flex flex-col gap-3 overflow-y-auto custom-scrollbar px-1">
            <div className="shrink-0 space-y-1">
              <div className="text-[9px] font-black uppercase text-slate-400 tracking-widest ml-2">Source (Mandarin)</div>
              <div className="bg-white dark:bg-neutral-900 border border-slate-200 dark:border-neutral-800 p-4 rounded-xl text-base leading-loose text-slate-600 dark:text-neutral-300 font-sans shadow-sm h-auto max-h-[250px] overflow-y-auto custom-scrollbar">{currentData?.src}</div>
            </div>
            <div className="shrink-0 space-y-1">
              <div className="flex justify-between px-2"><div className="text-[9px] font-black uppercase text-blue-500 tracking-widest">Translation (Taiwanese)</div><div className="text-[9px] text-slate-400 uppercase tracking-tighter">Select text to mark errors</div></div>
              <div ref={tgtRef} onMouseUp={handleSelection} onClick={() => setSelectedSpanId(null)} className="h-auto max-h-[250px] bg-white dark:bg-neutral-900 border-2 border-blue-50 dark:border-blue-900/20 p-4 rounded-xl text-base leading-loose shadow-sm overflow-y-auto selection:bg-blue-200 dark:selection:bg-blue-500/30 font-sans">{renderText()}</div>
            </div>
          </div>

          {/* Explanation & Scoring Area - Optimized Spacing */}
          <div className="grid grid-cols-2 gap-4 shrink-0 h-40 mb-1">
            <div className="flex flex-col gap-1.5">
              <div className="text-[9px] font-black uppercase text-slate-400 tracking-widest ml-2 flex items-center gap-1.5"><Info size={12} className="text-blue-500" /> Explanation</div>
              <textarea placeholder="Provide linguistic justification..." className="flex-1 bg-white dark:bg-neutral-900 border border-slate-200 dark:border-neutral-800 rounded-xl p-3 text-xs font-sans outline-none focus:ring-1 focus:ring-blue-500 transition-all resize-none shadow-sm custom-scrollbar placeholder:text-slate-400 dark:placeholder:text-neutral-600"
                value={explanation} onChange={e => setExplanation(e.target.value)} />
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="text-[9px] font-black uppercase text-slate-400 tracking-widest ml-2 flex items-center gap-1.5"><CheckCircle2 size={12} className="text-green-500" /> Score (0.0 - 6.0)</div>
              <div className="flex-1 bg-white dark:bg-neutral-900 border border-slate-200 dark:border-neutral-800 rounded-xl p-4 shadow-sm flex flex-col justify-center gap-2">
                {/* Score Labels - Corrected Alignment */}
                <div className="relative h-6 w-full mx-1">
                  {[{ p: '0%', l: 'No meaning', pos: '0%' }, { p: '33%', l: 'Some', pos: '33.33%' }, { p: '66%', l: 'Most', pos: '66.66%' }, { p: '100%', l: 'Perfect', pos: '100%' }].map((t, i, arr) => (
                    <div key={i} className="absolute top-0 flex flex-col items-center"
                      style={{
                        left: t.pos,
                        transform: i === 0 ? 'translateX(0)' : i === arr.length - 1 ? 'translateX(-100%)' : 'translateX(-50%)',
                        alignItems: i === 0 ? 'flex-start' : i === arr.length - 1 ? 'flex-end' : 'center'
                      }}>
                      <span className="text-[8px] font-black text-slate-400 mb-0.5">{t.p}</span>
                      <span className={`text-[9px] font-bold text-slate-500 w-16 leading-tight uppercase tracking-tighter ${i === 0 ? 'text-left' : i === arr.length - 1 ? 'text-right' : 'text-center'}`}>{t.l}</span>
                    </div>
                  ))}
                </div>
                {/* Custom Gradient Slider */}
                <div className="relative px-2">
                  <input type="range" min="0" max="6" step="0.1" value={score7} onChange={e => setScore7(parseFloat(e.target.value))} className="annotation-slider w-full h-2 bg-slate-100 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer" />
                  {/* Tick Marks Alignment */}
                  <div className="absolute top-0 left-2.5 right-2.5 h-2 flex justify-between pointer-events-none px-0.5 opacity-20">
                    {[0, 1, 2, 3].map(i => <div key={i} className="w-0.5 h-full bg-slate-400 dark:bg-slate-300"></div>)}
                  </div>
                </div>
                <div className="flex justify-center -mt-1"><div className="px-3 py-0.5 bg-slate-100 dark:bg-neutral-800 text-slate-700 dark:text-slate-300 rounded-lg font-bold text-sm border border-slate-200 dark:border-neutral-700 font-mono shadow-sm">{score7.toFixed(1)}</div></div>
              </div>
            </div>
          </div>
        </section>

        {/* Right Sidebar - Error Inventory & Logs */}
        <aside className="w-80 shrink-0 bg-white dark:bg-neutral-900 border-l border-slate-200 dark:border-neutral-800 flex flex-col shadow-2xl z-20">
          {/* Error List */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="p-3 border-b border-slate-100 dark:border-neutral-800 flex items-center gap-2 font-black text-[9px] text-slate-500 uppercase bg-slate-50/50 dark:bg-neutral-800/20 tracking-widest">
              <AlertCircle size={14} className="text-orange-500" /> Error List ({spans.length})
            </div>
            <div className="flex-1 overflow-y-auto p-2 pt-3 space-y-3 custom-scrollbar bg-slate-50/10 dark:bg-transparent">
              {spans.length === 0 ? <div className="h-full flex flex-col items-center justify-center text-slate-300 space-y-2 opacity-50"><AlertTriangle size={24} strokeWidth={1} /><p className="text-[10px] font-bold uppercase tracking-widest">Clear</p></div>
                : spans.map(s => (
                  <div key={s.id}
                    ref={el => { spanRefs.current[s.id] = el; }}
                    onClick={() => setSelectedSpanId(s.id)}
                    className={`bg-white dark:bg-neutral-800 border rounded-xl p-2.5 shadow-sm transition-all cursor-pointer ${s.id === selectedSpanId ? 'border-blue-500 ring-1 ring-blue-500 bg-blue-50/10' : 'border-slate-200 dark:border-neutral-700 hover:border-blue-300 dark:hover:border-blue-700'}`}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-xs font-bold px-1.5 py-0.5 bg-slate-50 dark:bg-neutral-900 rounded text-blue-600 font-mono break-words border border-slate-100 dark:border-neutral-700 group relative cursor-help" title={s.text}>{s.text}</span>
                      <button onClick={(e) => { e.stopPropagation(); removeSpan(s.id); }} className="text-slate-300 hover:text-red-500 transition-colors p-1"><Trash2 size={12} /></button>
                    </div>
                    <div className="flex gap-1 mb-2">
                      {SEVERITIES.map(sev => (
                        <button key={sev.value} onClick={() => updateSeverity(s.id, sev.value)} className={`flex-1 text-[9px] font-black py-1 rounded transition-all border ${s.severity === sev.value ? 'bg-blue-600 text-white border-blue-600 shadow-md shadow-blue-500/20' : 'border-slate-200 dark:border-neutral-700 text-slate-400'}`}>{sev.label}</button>
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-1.5 pt-2 border-t border-slate-50 dark:border-neutral-700">
                      {/* Resize Controls */}
                      <div className="space-y-1">
                        <div className="text-[8px] font-bold text-slate-400 uppercase flex justify-between px-1"><span>L</span><span>RESIZE</span><span>R</span></div>
                        <div className="flex justify-between bg-slate-50 dark:bg-neutral-900 rounded p-0.5 border border-slate-100 dark:border-neutral-700">
                          <div className="flex"><button onClick={() => resizeSpan(s.id, 'start', -1)} className="p-1 hover:bg-white dark:hover:bg-neutral-700 rounded"><Minus size={10} /></button><button onClick={() => resizeSpan(s.id, 'start', 1)} className="p-1 hover:bg-white dark:hover:bg-neutral-700 rounded"><Plus size={10} /></button></div>
                          <div className="w-px bg-slate-200 dark:bg-neutral-800 my-0.5"></div>
                          <div className="flex"><button onClick={() => resizeSpan(s.id, 'end', -1)} className="p-1 hover:bg-white dark:hover:bg-neutral-700 rounded"><Minus size={10} /></button><button onClick={() => resizeSpan(s.id, 'end', 1)} className="p-1 hover:bg-white dark:hover:bg-neutral-700 rounded"><Plus size={10} /></button></div>
                        </div>
                      </div>
                      {/* Move Controls */}
                      <div className="space-y-1">
                        <div className="text-[8px] font-bold text-slate-400 uppercase text-center">MOVE</div>
                        <div className="flex justify-around bg-slate-50 dark:bg-neutral-900 rounded p-0.5 border border-slate-100 dark:border-neutral-700">
                          <button onClick={() => moveSpan(s.id, -1)} className="p-1 flex-1 flex justify-center hover:bg-white dark:hover:bg-neutral-700 rounded"><ChevronLeft size={10} /></button>
                          <button onClick={() => moveSpan(s.id, 1)} className="p-1 flex-1 flex justify-center hover:bg-white dark:hover:bg-neutral-700 rounded"><ChevronRight size={10} /></button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Action Logs */}
          <div className="h-32 border-t border-slate-200 dark:border-neutral-800 bg-slate-50/50 dark:bg-neutral-900/50 p-3 flex flex-col">
            <div className="flex items-center gap-2 text-[9px] font-black text-slate-400 uppercase mb-2 tracking-widest"><History size={12} /> Recent Actions</div>
            <div className="flex-1 overflow-y-auto space-y-1 font-mono text-[9px] custom-scrollbar">
              {logs.map(log => (
                <div key={log.id} className="flex gap-2 text-slate-500 items-center opacity-80 hover:opacity-100 transition-opacity">
                  <span className="opacity-40 shrink-0 uppercase text-[8px] tracking-tighter">{log.timestamp}</span>
                  <span className={`font-black uppercase text-[8px] w-8 ${log.action === 'Add' ? 'text-green-500' : log.action === 'Remove' ? 'text-red-500' : 'text-blue-500'}`}>{log.action}</span>
                  <span className="truncate opacity-80 text-slate-600 dark:text-neutral-400 leading-none">{log.details}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 3px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(156, 163, 175, 0.2); border-radius: 10px; }
        
        /* Range Input 美化 - Gradient track and sleek thumb */
        /* Range Input 美化 - Simply clean */
        .annotation-slider::-webkit-slider-runnable-track {
          background: #e2e8f0;
          height: 4px;
          border-radius: 999px;
        }
        .dark .annotation-slider::-webkit-slider-runnable-track {
          background: #334155;
        }
        .annotation-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          height: 14px;
          width: 14px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          margin-top: -5px;
          transition: transform 0.1s ease;
        }
        .annotation-slider:active::-webkit-slider-thumb {
          transform: scale(1.1);
        }
        .dark .annotation-slider::-webkit-slider-thumb {
          border-color: #171717;
        }

        /* Hide Number Input Spin Buttons */
        input::-webkit-outer-spin-button,
        input::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
        input[type=number] {
          -moz-appearance: textfield;
        }
      `}</style>
    </div>
  );
}