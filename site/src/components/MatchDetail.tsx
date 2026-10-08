import { useEffect, useRef } from "react";
import { OUTCOME_LABEL, OUTCOMES, kickoffDay, kickoffTime, lead, pct, signed, utcStamp } from "../format";
import type { Match } from "../types";

interface Props {
  match: Match;
  repo: string;
  onClose: () => void;
}

function form(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

/** Side panel on wide screens, bottom sheet on phones (see .detail in styles.css). */
export function MatchDetail({ match, repo, onClose }: Props) {
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [match.id, onClose]);

  const title = match.score
    ? `${match.home} ${match.score[0]}–${match.score[1]} ${match.away}`
    : `${match.home} v ${match.away}`;
  const settled = match.result !== null;

  return (
    <div className="detail">
      <div className="detail__backdrop" onClick={onClose} aria-hidden="true" />
      <section className="card detail__panel" role="dialog" aria-modal="false" aria-labelledby="detail-title">
        <span className="detail__grip" aria-hidden="true" />
        <div className="detail__head">
          <div>
            <h2 id="detail-title" className="detail__title">
              {title}
            </h2>
            <p className="muted">
              {kickoffDay(match.kickoff)}, {kickoffTime(match.kickoff)} UK
              {settled && ` · result: ${OUTCOME_LABEL[match.result!]}`}
            </p>
          </div>
          <button ref={closeButton} type="button" className="icon-button" onClick={onClose} aria-label="Close match detail">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        {match.source === "live" ? (
          <p className="provenance provenance--live">
            Logged {match.loggedAt && utcStamp(match.loggedAt)}, {lead(match)} before kick-off
            {match.commit && (
              <>
                {" · "}
                <a href={`${repo}/commit/${match.commit}`}>commit {match.commit.slice(0, 7)}</a>
              </>
            )}
          </p>
        ) : (
          <p className="provenance provenance--simulated">
            Simulated: computed after the match with the frozen model, from the matches played before it.
            Not part of the live record.
          </p>
        )}

        <table className="detail__table">
          <thead>
            <tr>
              <th scope="col">Outcome</th>
              <th scope="col">Model</th>
              <th scope="col">Bet365</th>
              <th scope="col">Gap</th>
            </tr>
          </thead>
          <tbody>
            {OUTCOMES.map((o) => (
              <tr key={o} className={match.result === o ? "is-result" : undefined}>
                <th scope="row">
                  <span className={`swatch swatch--${o}`} aria-hidden="true" />
                  {OUTCOME_LABEL[o]}
                  {match.result === o && <span className="visually-hidden"> (result)</span>}
                </th>
                <td>
                  <strong>{pct(match.model[o])}</strong>
                </td>
                <td>{match.bookmaker ? pct(match.bookmaker[o]) : "—"}</td>
                <td>{match.bookmaker ? `${signed((match.model[o] - match.bookmaker[o]) * 100)} pts` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div>
          <h3 className="eyebrow">What the model saw before kick-off</h3>
          <table className="detail__features">
            <thead>
              <tr>
                <td />
                <th scope="col">{match.homeCode}</th>
                <th scope="col">{match.awayCode}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Elo rating</th>
                <td>{match.features.homeElo}</td>
                <td>{match.features.awayElo}</td>
              </tr>
              <tr>
                <th scope="row">Points per game, last 5</th>
                <td>{form(match.features.homeForm)}</td>
                <td>{form(match.features.awayForm)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {settled && (
          <div className="detail__verdicts">
            <p>
              Model picked <strong>{OUTCOME_LABEL[match.pick]}</strong> ·{" "}
              {match.pick === match.result ? <span className="chip chip--hit">right</span> : <span className="chip chip--miss">wrong</span>}
            </p>
            {match.bookmakerPick && (
              <p>
                Bet365 favourite <strong>{OUTCOME_LABEL[match.bookmakerPick]}</strong> ·{" "}
                {match.bookmakerPick === match.result ? (
                  <span className="chip chip--hit">right</span>
                ) : (
                  <span className="chip chip--miss">wrong</span>
                )}
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
