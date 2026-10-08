import { useEffect, useRef, useState, type PointerEvent, type RefObject } from "react";
import { pct, signed } from "../format";
import type { CalibrationBand } from "../types";

/** Width of an element in CSS px, kept current, so charts draw at their real size. */
function useWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry!.contentRect.width)));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

function niceStep(raw: number): number {
  const steps = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200];
  return steps.find((s) => s >= raw) ?? raw;
}

function ticks(lo: number, hi: number, count: number): number[] {
  const step = niceStep((hi - lo) / count);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/** Index of the nearest point to the pointer along the x axis. */
function nearest(event: PointerEvent<SVGElement>, left: number, plotWidth: number, n: number): number {
  const box = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - box.left - left;
  return Math.max(0, Math.min(n - 1, Math.round((x / plotWidth) * (n - 1))));
}

interface GapProps {
  values: number[];
  /** First match to draw: the opening weeks swing wildly on tiny samples. */
  from?: number;
  xLabels: { index: number; label: string }[];
  describe: (index: number) => string;
  ariaLabel: string;
}

/** Running log-loss gap to Bet365: above zero, the bookmaker was better. */
export function GapChart({ values: all, from = 0, xLabels: allLabels, describe: describeAt, ariaLabel }: GapProps) {
  const start = Math.min(from, Math.max(all.length - 1, 0));
  const values = all.slice(start);
  const xLabels = allLabels.filter((l) => l.index >= start).map((l) => ({ ...l, index: l.index - start }));
  const describe = (i: number) => describeAt(i + start);
  const [box, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = 240;
  const m = { top: 14, right: 16, bottom: 30, left: 52 };
  const w = Math.max(width - m.left - m.right, 10);
  const h = height - m.top - m.bottom;
  const lo = Math.min(-0.01, ...values) * 1.15;
  const hi = Math.max(0.02, ...values) * 1.1;
  const x = (i: number) => (values.length > 1 ? (i / (values.length - 1)) * w : w / 2);
  const y = (v: number) => ((hi - v) / (hi - lo)) * h;
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const last = values[values.length - 1] ?? 0;

  return (
    <div ref={box} className="chart">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          onPointerMove={(e) => setHover(nearest(e, m.left, w, values.length))}
          onPointerLeave={() => setHover(null)}
        >
          <g transform={`translate(${m.left},${m.top})`}>
            {ticks(lo, hi, 4).map((t) => (
              <g key={t}>
                <line x1={0} x2={w} y1={y(t)} y2={y(t)} className={t === 0 ? "axis-zero" : "grid"} />
                <text x={-10} y={y(t) + 4} textAnchor="end" className="tick">
                  {t === 0 ? "0" : signed(t, 2)}
                </text>
              </g>
            ))}
            <text x={6} y={y(0) + 16} className="tick">
              Level with Bet365
            </text>
            {xLabels.map(({ index, label }) => (
              <text key={index} x={x(index)} y={h + 22} className="tick" textAnchor={index === 0 ? "start" : "middle"}>
                {label}
              </text>
            ))}
            <path d={path} className="line line--model" />
            <circle cx={x(values.length - 1)} cy={y(last)} r={4.5} className="dot dot--model" />
            <text x={x(values.length - 1) - 6} y={y(last) - 12} textAnchor="end" className="annotation">
              {signed(last, 3)} after {all.length} {all.length === 1 ? "match" : "matches"}
            </text>
            {hover !== null && (
              <g>
                <line x1={x(hover)} x2={x(hover)} y1={0} y2={h} className="crosshair" />
                <circle cx={x(hover)} cy={y(values[hover]!)} r={5} className="dot dot--model" />
              </g>
            )}
          </g>
        </svg>
      )}
      {hover !== null && width > 0 && (
        <div
          className="tooltip"
          style={{ left: Math.min(m.left + x(hover) + 12, width - 190), top: m.top + 8 }}
        >
          <strong>{signed(values[hover]!, 3)}</strong> {describe(hover)}
        </div>
      )}
    </div>
  );
}

/** Stated confidence against how often the pick was right, one dot per band. */
export function CalibrationChart({ bands }: { bands: CalibrationBand[] }) {
  const [box, width] = useWidth<HTMLDivElement>();
  const size = Math.min(width, 340);
  const m = 40;
  const p = size - m - 12;
  const lo = 0.25;
  const hi = 0.85;
  const at = (v: number) => ((v - lo) / (hi - lo)) * p;
  const maxN = Math.max(...bands.map((b) => b.matches));
  return (
    <div ref={box} className="chart">
      {width > 0 && (
        <svg width={size} height={size} role="img" aria-label="Calibration: each confidence band's stated probability against how often its picks were right">
          <g transform={`translate(${m},12)`}>
            {[0.3, 0.5, 0.7].map((t) => (
              <g key={t}>
                <line x1={0} x2={p} y1={p - at(t)} y2={p - at(t)} className="grid" />
                <text x={-8} y={p - at(t) + 4} textAnchor="end" className="tick">
                  {pct(t)}
                </text>
                <text x={at(t)} y={p + 18} textAnchor="middle" className="tick">
                  {pct(t)}
                </text>
              </g>
            ))}
            <line x1={0} y1={p} x2={p} y2={0} className="axis-zero" />
            <text x={p - 4} y={14} textAnchor="end" className="tick">
              perfectly calibrated
            </text>
            {bands.map((b) => (
              <circle
                key={b.low}
                cx={at(b.stated)}
                cy={p - at(b.actual)}
                r={5 + 7 * Math.sqrt(b.matches / maxN)}
                className="dot dot--model dot--soft"
              >
                <title>{`Stated ${pct(b.stated)}, right ${pct(b.actual)} of ${b.matches} matches`}</title>
              </circle>
            ))}
          </g>
        </svg>
      )}
    </div>
  );
}

interface EloProps {
  paths: { name: string; path: number[] }[];
  selected: string;
}

/** Every club's rating this season in grey, the selected one on top. */
export function EloSeasonChart({ paths, selected }: EloProps) {
  const [box, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const height = 280;
  const m = { top: 12, right: 16, bottom: 30, left: 48 };
  const w = Math.max(width - m.left - m.right, 10);
  const h = height - m.top - m.bottom;
  const all = paths.flatMap((p) => p.path);
  const lo = Math.min(...all) - 15;
  const hi = Math.max(...all) + 15;
  const n = Math.max(...paths.map((p) => p.path.length));
  const x = (i: number) => (n > 1 ? (i / (n - 1)) * w : 0);
  const y = (v: number) => ((hi - v) / (hi - lo)) * h;
  const line = (path: number[]) => path.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const chosen = paths.find((p) => p.name === selected)!;
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(w / 44))));
  const shown = hover !== null ? Math.min(hover, chosen.path.length - 1) : null;

  return (
    <div ref={box} className="chart">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${selected}'s Elo rating after each match this season, against the other clubs`}
          onPointerMove={(e) => setHover(nearest(e, m.left, w, n))}
          onPointerLeave={() => setHover(null)}
        >
          <g transform={`translate(${m.left},${m.top})`}>
            {ticks(lo, hi, 5).map((t) => (
              <g key={t}>
                <line x1={0} x2={w} y1={y(t)} y2={y(t)} className={t === 1500 ? "axis-zero" : "grid"} />
                <text x={-8} y={y(t) + 4} textAnchor="end" className="tick">
                  {t}
                </text>
              </g>
            ))}
            {Array.from({ length: n }, (_, i) => i)
              .filter((i) => i === 0 || i === n - 1 || i % labelEvery === 0)
              .map((i) => (
                <text key={i} x={x(i)} y={h + 22} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"} className="tick">
                  {i === 0 ? "Start" : i}
                </text>
              ))}
            {paths
              .filter((p) => p.name !== selected)
              .map((p) => (
                <path key={p.name} d={line(p.path)} className="line line--context" />
              ))}
            <path d={line(chosen.path)} className="line line--focus" />
            {shown !== null && (
              <g>
                <line x1={x(shown)} x2={x(shown)} y1={0} y2={h} className="crosshair" />
                <circle cx={x(shown)} cy={y(chosen.path[shown]!)} r={5} className="dot dot--focus" />
              </g>
            )}
          </g>
        </svg>
      )}
      {shown !== null && width > 0 && (
        <div className="tooltip" style={{ left: Math.min(m.left + x(shown) + 12, width - 170), top: m.top + 8 }}>
          <strong>{chosen.path[shown]}</strong> {shown === 0 ? "at the start of the season" : `after match ${shown}`}
        </div>
      )}
    </div>
  );
}
