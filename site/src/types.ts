// Shapes of the JSON written by export_site.py.

export type Outcome = "H" | "D" | "A";
export type Probabilities = Record<Outcome, number>;

export interface Match {
  id: string;
  matchweek: number;
  kickoff: string; // UTC, "2026-10-17T11:30Z"
  home: string;
  away: string;
  homeCode: string;
  awayCode: string;
  score: [number, number] | null;
  result: Outcome | null;
  /** live: committed before kick-off. simulated: the frozen model run after the match. */
  source: "live" | "simulated";
  model: Probabilities;
  bookmaker: Probabilities | null;
  pick: Outcome;
  bookmakerPick: Outcome | null;
  loggedAt: string | null;
  commit: string | null;
  features: {
    homeElo: number;
    awayElo: number;
    homeForm: number | null;
    awayForm: number | null;
  };
}

export interface Score {
  accuracy: number;
  log_loss: number;
}

export interface SeasonRecord {
  matches: number;
  model?: Score;
  bookmaker?: Score;
  modelHits?: number;
  bookmakerHits?: number;
  gap?: number[];
}

export interface CalibrationBand {
  low: number;
  high: number;
  stated: number;
  actual: number;
  matches: number;
}

export interface Backtest {
  season: string;
  matches: number;
  model: Score;
  bookmaker: Score;
  baselines: { random: number; home: number; bookmaker: number; model: number };
  draws: { happened: number; called: number };
  gap: number[];
  dates: string[];
  calibration: CalibrationBand[];
}

export interface Record_ {
  backtest: Backtest;
  live: SeasonRecord;
  simulated: SeasonRecord;
}

export interface Team {
  name: string;
  code: string;
  rank: number;
  elo: number;
  start: number;
  change: number;
  path: number[];
  form: ("W" | "D" | "L")[];
}

export interface Meta {
  generatedAt: string;
  season: string;
  repo: string;
  model: { algorithm: string; features: number; version: string; fittedOn: string };
  ledger: { logged: number; settled: number; pending: number };
}

export interface SiteData {
  meta: Meta;
  matches: Match[];
  record: Record_;
  teams: Team[];
}
