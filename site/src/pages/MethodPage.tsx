import type { SiteData } from "../types";

const STEPS = [
  ["Fetch", "Results and fixtures", "From Football-Data.co.uk, retried when the site is busy."],
  ["Features", "Elo and recent form", "Only from matches played before kick-off. Odds are never inputs."],
  ["Forecast", "A frozen model", "Each row is checked: probabilities in range, summing to one, logged before kick-off."],
  ["Commit", "An append-only ledger", "The commit timestamps the forecast; the run fails if a logged row changes."],
] as const;

export function MethodPage({ data }: { data: SiteData }) {
  const { meta } = data;
  return (
    <div className="page">
      <div className="page-head">
        <div className="lead">
          <h1 className="display">How it works</h1>
          <p>
            Every Premier League match is forecast from public match history before kick-off, committed to the
            repository, then scored against Bet365 once the result is in. Nothing here is computed in your
            browser: the site is rebuilt from the repository after each run.
          </p>
        </div>
      </div>

      <section className="card section-card" aria-labelledby="pipeline-title">
        <h2 id="pipeline-title" className="display display--small">
          Every four hours
        </h2>
        <ol className="pipeline">
          {STEPS.map(([step, title, body], i) => (
            <li key={step} className="pipeline__step">
              <span className="eyebrow">
                {i + 1} · {step}
              </span>
              <span className="pipeline__title">{title}</span>
              <span>{body}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="card section-card" aria-labelledby="sources-title">
        <h2 id="sources-title" className="display display--small">
          Live and simulated forecasts
        </h2>
        <div className="sources">
          <div className="sources__item">
            <span className="badge badge--live">Logged before kick-off</span>
            <p>
              Made by the scheduled job before the match and committed to the ledger. The commit proves when it was
              made, and the row can never change afterwards. Only these count in the live record.
            </p>
          </div>
          <div className="sources__item">
            <span className="badge badge--simulated">Simulated</span>
            <p>
              Matches of {meta.season} played before the ledger started. The same frozen model, run after the match
              on the matches played before it: honest in method, but nobody can check when it was made, so these
              stay out of the live record.
            </p>
          </div>
        </div>
      </section>

      <section className="card section-card" aria-labelledby="validation-title">
        <h2 id="validation-title" className="display display--small">
          Validated in time order
        </h2>
        <div className="timeline" role="img" aria-label="2014/15 warm-up, 2015/16 to 2023/24 training, 2024/25 model choice, 2025/26 test, 2026/27 live">
          <span className="timeline__seg timeline__seg--warm" style={{ flexGrow: 1 }}>Warm-up</span>
          <span className="timeline__seg timeline__seg--train" style={{ flexGrow: 9 }}>Train · 2015/16 – 2023/24</span>
          <span className="timeline__seg timeline__seg--choose" style={{ flexGrow: 1 }}>Choose</span>
          <span className="timeline__seg timeline__seg--test" style={{ flexGrow: 1 }}>Test</span>
          <span className="timeline__seg timeline__seg--live" style={{ flexGrow: 1 }}>Live</span>
        </div>
        <p className="timeline__ends muted">
          <span>2014/15</span>
          <span>2024/25 · 2025/26 · {meta.season}</span>
        </p>
        <p>
          Candidates are compared on 2024/25; the winner is refitted and scored once on 2025/26. For the live season
          the {meta.model.algorithm.toLowerCase()} is refitted on {meta.model.fittedOn} and frozen until the summer
          (version {meta.model.version}, {meta.model.features} pre-match features).
        </p>
      </section>

      <div className="three-col">
        <section className="card section-card">
          <h2 className="display display--small">Guarded by tests</h2>
          <ul className="bullets">
            <li>Changing a result never changes that match's own features, or any earlier ones</li>
            <li>The live code gives the same probabilities as the training code for the same match</li>
            <li>Logged rows are never rewritten, and invalid rows are refused</li>
          </ul>
        </section>
        <section className="card section-card">
          <h2 className="display display--small">Limitations</h2>
          <ul className="bullets">
            <li>No line-ups, injuries or expected goals: that is where bookmakers get their edge</li>
            <li>Draws are almost never the single likeliest outcome, so they are rarely picked</li>
            <li>Clubs promoted after years away start from an old rating</li>
          </ul>
        </section>
        <section className="card section-card section-card--dark">
          <h2 className="display display--small">Look inside</h2>
          <ul className="links">
            <li>
              <a href={meta.repo}>Source code on GitHub</a>
            </li>
            <li>
              <a href={`${meta.repo}/commits/main/predictions/live.csv`}>Forecast ledger history</a>
            </li>
            <li>
              <a href="https://www.football-data.co.uk/englandm.php">Data: Football-Data.co.uk</a>
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
