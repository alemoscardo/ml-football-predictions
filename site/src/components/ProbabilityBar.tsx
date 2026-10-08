import { OUTCOME_LABEL, OUTCOMES, pct } from "../format";
import type { Probabilities } from "../types";

interface Props {
  model: Probabilities;
  bookmaker: Probabilities | null;
  size?: "regular" | "compact";
}

/** The model's home / draw / away split, with Bet365's as a thin bar underneath. */
export function ProbabilityBar({ model, bookmaker, size = "regular" }: Props) {
  const summary = OUTCOMES.map((o) => `${OUTCOME_LABEL[o]} ${pct(model[o])}`).join(", ");
  return (
    <span className={`prob prob--${size}`}>
      <span className="prob__model" role="img" aria-label={`Model: ${summary}`}>
        {OUTCOMES.map((o) => (
          <span key={o} className={`prob__seg prob__seg--${o}`} style={{ flexGrow: model[o] }}>
            {model[o] >= 0.12 ? pct(model[o]) : ""}
          </span>
        ))}
      </span>
      {bookmaker && (
        <span
          className="prob__bookie"
          role="img"
          aria-label={`Bet365: ${OUTCOMES.map((o) => pct(bookmaker[o])).join(", ")}`}
        >
          {OUTCOMES.map((o) => (
            <span
              key={o}
              className={`prob__thin prob__thin--${o}`}
              style={{ flexGrow: bookmaker[o] }}
            />
          ))}
        </span>
      )}
    </span>
  );
}
