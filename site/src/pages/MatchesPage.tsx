import { useCallback, useMemo, useRef, useState } from "react";
import { MatchCard } from "../components/MatchCard";
import { MatchDetail } from "../components/MatchDetail";
import { dateRange, isHit, isSettled } from "../format";
import { go, href } from "../route";
import type { Match, SiteData } from "../types";

type Filter = "all" | "hit" | "miss";
const FILTERS: [Filter, string][] = [
  ["all", "All"],
  ["hit", "Correct"],
  ["miss", "Missed"],
];
const NEXT = "next";

function defaultWeek(matches: Match[], weeks: number[]): number | typeof NEXT {
  const pending = matches.filter((m) => !isSettled(m));
  if (pending.length) return Math.min(...pending.map((m) => m.matchweek));
  return weeks.length ? Math.max(...weeks) : NEXT;
}

export function MatchesPage({ data, param }: { data: SiteData; param: string | null }) {
  const { matches, teams, meta } = data;
  const weeks = useMemo(() => [...new Set(matches.map((m) => m.matchweek))].sort((a, b) => a - b), [matches]);
  const hasPending = matches.some((m) => !isSettled(m));
  const requested = param === NEXT ? NEXT : Number(param);
  const week = (requested === NEXT && !hasPending) || weeks.includes(requested as number)
    ? requested
    : defaultWeek(matches, weeks);

  const [filter, setFilter] = useState<Filter>("all");
  const [openId, setOpenId] = useState<string | null>(null);
  const lastCard = useRef<HTMLButtonElement | null>(null);

  const inWeek = matches.filter((m) => m.matchweek === week);
  const settled = inWeek.filter(isSettled);
  const shown = inWeek.filter((m) =>
    filter === "all" ? true : isSettled(m) && (filter === "hit") === isHit(m),
  );
  const open = inWeek.find((m) => m.id === openId) ?? null;
  const hasSimulated = inWeek.some((m) => m.source === "simulated");

  const close = useCallback(() => {
    setOpenId(null);
    lastCard.current?.focus();
  }, []);

  const pick = (target: number | typeof NEXT) => {
    setFilter("all");
    setOpenId(null);
    go("matches", target);
  };

  const modelHits = settled.filter(isHit).length;
  const bookieHits = settled.filter((m) => m.bookmakerPick === m.result).length;
  const subtitle =
    week === NEXT
      ? "Fixtures not published yet"
      : [
          dateRange(inWeek),
          settled.length
            ? `the model called ${modelHits} of ${settled.length}, Bet365's favourite ${bookieHits}`
            : `${inWeek.length} forecasts logged, awaiting results`,
        ].join(" · ");

  const rounds = weeks
    .map((w) => {
      const done = matches.filter((m) => m.matchweek === w && isSettled(m));
      return {
        week: w,
        played: done.length,
        model: done.filter(isHit).length,
        bookie: done.filter((m) => m.bookmakerPick === m.result).length,
        simulated: done.some((m) => m.source === "simulated"),
      };
    })
    .filter((r) => r.played > 0);
  const total = rounds.reduce(
    (acc, r) => ({ played: acc.played + r.played, model: acc.model + r.model, bookie: acc.bookie + r.bookie }),
    { played: 0, model: 0, bookie: 0 },
  );

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="display">{week === NEXT ? "Next matchweek" : `Matchweek ${week}`}</h1>
          <p className="muted">{subtitle}</p>
        </div>
        <div className="week-picker" role="group" aria-label="Matchweek">
          {weeks.map((w) => (
            <button key={w} type="button" className="pill" aria-pressed={w === week} onClick={() => pick(w)}>
              {w}
            </button>
          ))}
          {!hasPending && (
            <button type="button" className="pill pill--next" aria-pressed={week === NEXT} onClick={() => pick(NEXT)}>
              Next
            </button>
          )}
        </div>
      </div>

      {week !== NEXT && (settled.length > 0 || hasSimulated) && (
        <div className="toolbar">
          {settled.length > 0 && (
            <div className="segmented" role="group" aria-label="Show">
              {FILTERS.map(([value, label]) => (
                <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>
                  {label}
                </button>
              ))}
            </div>
          )}
          {hasSimulated && (
            <p className="notice notice--simulated">
              Simulated · computed after the match with the frozen model, so not part of the live record.{" "}
              <a href={href("method")}>What's the difference?</a>
            </p>
          )}
        </div>
      )}

      <div className="with-aside">
        <section aria-label="Matches" className="main-col">
          {week === NEXT ? (
            <div className="card empty">
              <h2 className="display display--small">Forecasts coming</h2>
              <p>
                Football-Data publishes the next fixtures a few days before kick-off. Every four hours a GitHub
                Actions job checks, forecasts each match and commits it, so every card here will carry the time
                it was logged.
              </p>
              <a href={href("method")}>How the ledger works</a>
            </div>
          ) : shown.length ? (
            <div className="card-grid">
              {shown.map((m) => (
                <MatchCard
                  key={m.id}
                  match={m}
                  selected={m.id === openId}
                  onOpen={(match, card) => {
                    lastCard.current = card;
                    setOpenId(match.id === openId ? null : match.id);
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="card empty">No matches for this filter.</div>
          )}
        </section>

        <aside className="aside-col">
          {open && <MatchDetail key={open.id} match={open} repo={meta.repo} onClose={close} />}

          {rounds.length > 0 && (
            <section className="card side-card" aria-labelledby="rounds-title">
              <h2 id="rounds-title" className="display display--small">
                Round by round
              </h2>
              <table className="rounds">
                <thead>
                  <tr>
                    <th scope="col">Matchweek</th>
                    <th scope="col">Model</th>
                    <th scope="col">Bet365 fav.</th>
                  </tr>
                </thead>
                <tbody>
                  {rounds.map((r) => (
                    <tr key={r.week} className={r.week === week ? "is-current" : undefined}>
                      <th scope="row">
                        <a href={href("matches", r.week)}>MW {r.week}</a>
                        {r.simulated && <span className="tag">simulated</span>}
                      </th>
                      <td>
                        {r.model} / {r.played}
                      </td>
                      <td>
                        {r.bookie} / {r.played}
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <th scope="row">Total</th>
                    <td>
                      {total.model} / {total.played}
                    </td>
                    <td>
                      {total.bookie} / {total.played}
                    </td>
                  </tr>
                </tfoot>
              </table>
            </section>
          )}

          <section className="card side-card" aria-labelledby="elo-title">
            <h2 id="elo-title" className="display display--small">
              Elo top 5
            </h2>
            <ol className="top-list">
              {teams.slice(0, 5).map((t) => (
                <li key={t.code}>
                  <a href={href("teams", t.code)}>{t.name}</a>
                  <span className="figure">{t.elo}</span>
                </li>
              ))}
            </ol>
            <a href={href("teams")}>All {teams.length} clubs</a>
          </section>
        </aside>
      </div>
    </div>
  );
}
