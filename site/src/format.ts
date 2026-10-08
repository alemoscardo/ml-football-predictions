import type { Match, Outcome } from "./types";

export const OUTCOMES: Outcome[] = ["H", "D", "A"];
export const OUTCOME_LABEL: Record<Outcome, string> = { H: "Home win", D: "Draw", A: "Away win" };

const UK = "Europe/London";

export function pct(p: number, digits = 0): string {
  return `${(p * 100).toFixed(digits)}%`;
}

/** Signed number with a real minus sign: +12, −3, 0. */
export function signed(n: number, digits = 0): string {
  const text = Math.abs(n).toFixed(digits);
  if (Number(text) === 0) return text;
  return `${n > 0 ? "+" : "−"}${text}`;
}

/** Recent ICU data abbreviates September as "Sept"; keep every month to three letters. */
function threeLetterMonths(text: string): string {
  return text.replace(/\bSept\b/, "Sep");
}

export function kickoffDay(iso: string): string {
  return threeLetterMonths(
    new Intl.DateTimeFormat("en-GB", {
      timeZone: UK,
      weekday: "short",
      day: "numeric",
      month: "short",
    }).format(new Date(iso)),
  );
}

export function kickoffTime(iso: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: UK,
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));
}

export function utcStamp(iso: string): string {
  const stamp = new Intl.DateTimeFormat("en-GB", {
    timeZone: "UTC",
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));
  return `${threeLetterMonths(stamp)} UTC`;
}

/** "2d 4h" between logging and kick-off. */
export function lead(match: Match): string | null {
  if (!match.loggedAt) return null;
  const hours = Math.floor((Date.parse(match.kickoff) - Date.parse(match.loggedAt)) / 3_600_000);
  const days = Math.floor(hours / 24);
  return days > 0 ? `${days}d ${hours % 24}h` : `${hours}h`;
}

export function dateRange(matches: Match[]): string {
  if (!matches.length) return "";
  const fmt = (iso: string, withMonth: boolean) =>
    new Intl.DateTimeFormat("en-GB", {
      timeZone: UK,
      day: "numeric",
      ...(withMonth ? { month: "long" } : {}),
    }).format(new Date(iso));
  const first = matches[0]!.kickoff;
  const last = matches[matches.length - 1]!.kickoff;
  const sameMonth = fmt(first, true).split(" ")[1] === fmt(last, true).split(" ")[1];
  if (fmt(first, true) === fmt(last, true)) return fmt(first, true);
  return sameMonth ? `${fmt(first, false)}–${fmt(last, true)}` : `${fmt(first, true)} – ${fmt(last, true)}`;
}

export function isSettled(match: Match): boolean {
  return match.result !== null;
}

export function isHit(match: Match): boolean {
  return match.result !== null && match.pick === match.result;
}
