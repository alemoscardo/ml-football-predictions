import { useEffect, useRef } from "react";
import { TeamBadge } from "../components/Badges";
import { EloSeasonChart } from "../components/charts";
import { signed } from "../format";
import { go, href } from "../route";
import type { SiteData, Team } from "../types";

const RESULT_WORD = { W: "Won", D: "Drew", L: "Lost" } as const;

function Form({ team, size }: { team: Team; size: "small" | "large" }) {
  return (
    <span className={`form form--${size}`} aria-label={`Last ${team.form.length}: ${team.form.map((r) => RESULT_WORD[r]).join(", ")}`}>
      {team.form.map((r, i) => (
        <span key={i} className={`form__chip form__chip--${r}`} aria-hidden="true">
          {size === "large" ? RESULT_WORD[r] : r}
        </span>
      ))}
    </span>
  );
}

export function TeamsPage({ data, param }: { data: SiteData; param: string | null }) {
  const { teams, meta } = data;
  const team = teams.find((t) => t.code === param) ?? teams[0];
  const detail = useRef<HTMLElement>(null);

  // On a phone the detail sits below the 20-row table: bring it into view on a pick.
  useEffect(() => {
    if (param && window.matchMedia("(max-width: 900px)").matches) {
      detail.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [param]);

  if (!team) return <div className="page">No club data yet.</div>;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="display">Teams</h1>
          <p className="muted">Elo rating after each club's latest match · 1500 is the long-run league average</p>
        </div>
      </div>

      <div className="with-aside with-aside--left">
        <section className="card ranking" aria-label="Elo ranking">
          <div className="ranking__head" aria-hidden="true">
            <span>#</span>
            <span>Club</span>
            <span>Form</span>
            <span>Elo</span>
            <span>Since Aug</span>
          </div>
          {teams.map((t) => (
            <button
              key={t.code}
              type="button"
              className="ranking__row"
              aria-pressed={t.code === team.code}
              onClick={() => go("teams", t.code)}
            >
              <span className="muted">{t.rank}</span>
              <span className="strong">{t.name}</span>
              <Form team={t} size="small" />
              <span className="figure">{t.elo}</span>
              <span>{signed(t.change)}</span>
            </button>
          ))}
        </section>

        <section ref={detail} className="card team-detail" aria-labelledby="team-title">
          <div className="team-detail__head">
            <div className="team-detail__name">
              <TeamBadge code={team.code} size={56} />
              <div>
                <h2 id="team-title" className="display">
                  {team.name}
                </h2>
                <p className="muted">
                  #{team.rank} of {teams.length} · started {meta.season} at {team.start}
                </p>
              </div>
            </div>
            <div className="team-detail__figures">
              <div>
                <span className="muted">Elo now</span>
                <span className="figure figure--large">{team.elo}</span>
              </div>
              <div>
                <span className="muted">Since August</span>
                <span className="figure figure--large">{signed(team.change)}</span>
              </div>
            </div>
          </div>
          <div>
            <h3 className="eyebrow">Results this season, oldest first</h3>
            <Form team={team} size="large" />
          </div>
          <figure className="figure-block">
            <figcaption className="eyebrow">Elo this season · {team.name} against the rest</figcaption>
            <EloSeasonChart paths={teams.map((t) => ({ name: t.name, path: t.path }))} selected={team.name} />
          </figure>
          <a href={href("matches")}>See the latest matchweek</a>
        </section>
      </div>
    </div>
  );
}
