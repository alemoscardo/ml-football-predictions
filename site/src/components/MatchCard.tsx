import { OUTCOME_LABEL, kickoffDay, kickoffTime } from "../format";
import type { Match } from "../types";
import { SourceBadge, TeamBadge, Verdict } from "./Badges";
import { ProbabilityBar } from "./ProbabilityBar";

interface Props {
  match: Match;
  selected: boolean;
  onOpen: (match: Match, card: HTMLButtonElement) => void;
}

export function MatchCard({ match, selected, onOpen }: Props) {
  const [homeGoals, awayGoals] = match.score ?? [null, null];
  const label = match.score
    ? `${match.home} ${homeGoals}, ${match.away} ${awayGoals}`
    : `${match.home} against ${match.away}`;
  return (
    <button
      type="button"
      className={`card match-card${selected ? " is-selected" : ""}`}
      aria-pressed={selected}
      aria-label={`${label}. Show forecast detail`}
      onClick={(event) => onOpen(match, event.currentTarget)}
    >
      <span className="match-card__top">
        <span>
          {kickoffDay(match.kickoff)} · {kickoffTime(match.kickoff)}
        </span>
        <SourceBadge match={match} />
      </span>
      <span className="match-card__teams">
        {[
          [match.homeCode, match.home, homeGoals],
          [match.awayCode, match.away, awayGoals],
        ].map(([code, name, goals]) => (
          <span className="match-card__team" key={String(name)}>
            <TeamBadge code={String(code)} />
            <span className="match-card__name">{name}</span>
            {goals !== null && <span className="match-card__goals">{goals}</span>}
          </span>
        ))}
      </span>
      <ProbabilityBar model={match.model} bookmaker={match.bookmaker} />
      <span className="match-card__foot">
        <span>
          Pick: <strong>{OUTCOME_LABEL[match.pick]}</strong>
        </span>
        <Verdict match={match} />
      </span>
    </button>
  );
}
