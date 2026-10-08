import { isHit, lead } from "../format";
import type { Match } from "../types";

/** Where a forecast comes from: committed before kick-off, or simulated after the match. */
export function SourceBadge({ match }: { match: Match }) {
  if (match.source === "simulated") {
    return (
      <span className="badge badge--simulated" title="Computed after the match with the frozen model: not part of the live record">
        Simulated
      </span>
    );
  }
  const before = lead(match);
  return (
    <span className="badge badge--live" title="Committed to the ledger before kick-off">
      {before ? `Logged ${before} before` : "Logged before kick-off"}
    </span>
  );
}

export function Verdict({ match }: { match: Match }) {
  if (match.result === null) return <span className="chip chip--pending">Awaiting result</span>;
  return isHit(match) ? (
    <span className="chip chip--hit">Correct</span>
  ) : (
    <span className="chip chip--miss">Missed</span>
  );
}

export function TeamBadge({ code, size = 34 }: { code: string; size?: number }) {
  return (
    <span className="team-badge" style={{ width: size, height: size }} aria-hidden="true">
      {code}
    </span>
  );
}
