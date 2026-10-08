import { CalibrationChart, GapChart } from "../components/charts";
import { pct } from "../format";
import { go, href } from "../route";
import type { SiteData } from "../types";

type View = "live" | "backtest";

function monthLabels(dates: string[]): { index: number; label: string }[] {
  const out: { index: number; label: string }[] = [];
  let previous = "";
  dates.forEach((date, index) => {
    const month = new Date(`${date}T12:00:00Z`).toLocaleString("en-GB", { month: "short", timeZone: "UTC" });
    if (month !== previous) out.push({ index, label: month });
    previous = month;
  });
  return out.filter((_, i) => i % 2 === 0); // every other month keeps phones legible
}

export function RecordPage({ data, param }: { data: SiteData; param: string | null }) {
  const { backtest, live, simulated } = data.record;
  const view: View = param === "live" || param === "backtest" ? param : live.matches > 0 ? "live" : "backtest";
  const views: [View, string][] = [
    ["live", `Live ${data.meta.season}`],
    ["backtest", `Backtest ${backtest.season}`],
  ];

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="display">Track record</h1>
          <p className="muted">How the model's forecasts compare with Bet365's</p>
        </div>
        <div className="segmented" role="group" aria-label="Record">
          {views.map(([value, label]) => (
            <button key={value} type="button" aria-pressed={view === value} onClick={() => go("record", value)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {view === "live" && live.matches === 0 && (
        <section className="card empty empty--split">
          <div>
            <h2 className="display display--small">The live record starts with the next round</h2>
            <p>
              Every match is forecast before kick-off and committed to an append-only ledger in the repository.
              Once results come in, this page scores those forecasts against Bet365, match by match, with a
              running gap like the backtest's.
            </p>
            <a href={`${data.meta.repo}/commits/main/predictions/live.csv`}>Ledger commit history</a>
          </div>
          {simulated.matches > 0 && (
            <div className="simulated-box">
              <p className="eyebrow eyebrow--simulated">Meanwhile · {data.meta.season} so far, simulated</p>
              <div className="simulated-box__figures">
                <div>
                  <span className="figure figure--large">
                    {simulated.modelHits} / {simulated.matches}
                  </span>
                  <span className="muted">model</span>
                </div>
                <div>
                  <span className="figure figure--large">
                    {simulated.bookmakerHits} / {simulated.matches}
                  </span>
                  <span className="muted">Bet365 favourite</span>
                </div>
              </div>
              <p className="small">Computed after kick-off with the frozen model, so not part of the live record.</p>
            </div>
          )}
        </section>
      )}

      {view === "live" && live.matches > 0 && live.model && live.bookmaker && (
        <>
          <div className="kpis">
            <div className="card kpi">
              <span className="muted">Accuracy</span>
              <span className="figure figure--large">{pct(live.model.accuracy, 1)}</span>
              <span className="muted">Bet365 favourite {pct(live.bookmaker.accuracy, 1)}</span>
            </div>
            <div className="card kpi">
              <span className="muted">Log-loss (lower is better)</span>
              <span className="figure figure--large">{live.model.log_loss.toFixed(3)}</span>
              <span className="muted">Bet365 {live.bookmaker.log_loss.toFixed(3)}</span>
            </div>
            <div className="card kpi">
              <span className="muted">Settled forecasts</span>
              <span className="figure figure--large">{live.matches}</span>
              <span className="muted">all logged before kick-off</span>
            </div>
          </div>
          <section className="card chart-card">
            <div className="chart-card__head">
              <h2 className="display display--small">Gap to Bet365 so far</h2>
              <p className="muted">Cumulative log-loss, model minus Bet365 · above zero, Bet365 ahead</p>
            </div>
            <GapChart
              values={live.gap ?? []}
              xLabels={[{ index: 0, label: "1st match" }]}
              describe={(i) => `after ${i + 1} live ${i ? "matches" : "match"}`}
              ariaLabel="Cumulative log-loss gap to Bet365 over the live forecasts"
            />
            <p className="small muted">
              A few hundred matches are needed to tell the two apart; read this as a check on the backtest.
            </p>
          </section>
        </>
      )}

      {view === "backtest" && (
        <>
          <div className="kpis">
            <div className="card kpi">
              <span className="muted">Accuracy</span>
              <span className="figure figure--large">{pct(backtest.model.accuracy, 1)}</span>
              <span className="muted">Bet365 favourite {pct(backtest.bookmaker.accuracy, 1)}</span>
            </div>
            <div className="card kpi">
              <span className="muted">Log-loss (lower is better)</span>
              <span className="figure figure--large">{backtest.model.log_loss.toFixed(3)}</span>
              <span className="muted">Bet365 {backtest.bookmaker.log_loss.toFixed(3)} · coin-flip 1.099</span>
            </div>
            <div className="card kpi">
              <span className="muted">Matches</span>
              <span className="figure figure--large">{backtest.matches}</span>
              <span className="muted">{backtest.season}, never seen in training</span>
            </div>
            <div className="card kpi">
              <span className="muted">Draws called</span>
              <span className="figure figure--large">
                {backtest.draws.called} of {backtest.draws.happened}
              </span>
              <span className="muted">Rarely the likeliest single outcome</span>
            </div>
          </div>

          <section className="card chart-card">
            <div className="chart-card__head">
              <h2 className="display display--small">Gap to Bet365 through the season</h2>
              <p className="muted">
                Cumulative log-loss, model minus Bet365, from the 20th match · above zero, Bet365 ahead
              </p>
            </div>
            <GapChart
              values={backtest.gap}
              from={19}
              xLabels={monthLabels(backtest.dates)}
              describe={(i) => `after ${i + 1} matches, ${new Date(`${backtest.dates[i]}T12:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" })}`}
              ariaLabel={`Cumulative log-loss gap to Bet365 over ${backtest.season}`}
            />
          </section>

          <div className="two-col">
            <section className="card chart-card">
              <h2 className="display display--small">Against simple baselines</h2>
              <ul className="baselines">
                {(
                  [
                    ["Random guess", backtest.baselines.random, false],
                    ["Always back the home side", backtest.baselines.home, false],
                    ["Bet365 favourite", backtest.baselines.bookmaker, false],
                    ["Model", backtest.baselines.model, true],
                  ] as const
                ).map(([label, value, isModel]) => (
                  <li key={label} className={isModel ? "is-model" : undefined}>
                    <span className="baselines__label">
                      <span>{label}</span>
                      <strong>{pct(value, 1)}</strong>
                    </span>
                    <span className="baselines__track">
                      <span className="baselines__bar" style={{ width: pct(value, 1) }} />
                    </span>
                  </li>
                ))}
              </ul>
              <p className="small muted">Share of the {backtest.matches} matches called correctly</p>
            </section>
            <section className="card chart-card">
              <h2 className="display display--small">Calibration</h2>
              <p className="muted">When it says 55%, is it right 55% of the time?</p>
              <CalibrationChart bands={backtest.calibration} />
              <p className="small muted">x: stated confidence · y: share correct · dot size: matches in the band</p>
            </section>
          </div>
          <p className="small muted">
            Method and validation: <a href={href("method")}>How it works</a>
          </p>
        </>
      )}
    </div>
  );
}
