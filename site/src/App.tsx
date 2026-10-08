import { useEffect, useState } from "react";
import { loadSiteData } from "./data";
import { utcStamp } from "./format";
import { MatchesPage } from "./pages/MatchesPage";
import { MethodPage } from "./pages/MethodPage";
import { RecordPage } from "./pages/RecordPage";
import { TeamsPage } from "./pages/TeamsPage";
import { href, useRoute, type Page } from "./route";
import type { SiteData } from "./types";

const NAV: [Page, string][] = [
  ["matches", "Matches"],
  ["record", "Record"],
  ["teams", "Teams"],
  ["method", "Method"],
];

export function App() {
  const route = useRoute();
  const [data, setData] = useState<SiteData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSiteData().then(setData, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    const titles: Record<Page, string> = { matches: "Matches", record: "Track record", teams: "Teams", method: "How it works" };
    document.title = `${titles[route.page]} · Premier League Outcome Model`;
    window.scrollTo({ top: 0 });
  }, [route.page]);

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <div className="site-header__inner">
          <a className="brand" href={href("matches")}>
            EPL Outcome Model
          </a>
          <nav aria-label="Main">
            <ul className="nav">
              {NAV.map(([page, label]) => (
                <li key={page}>
                  <a href={href(page)} aria-current={route.page === page ? "page" : undefined}>
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      <main id="main" tabIndex={-1}>
        {error && (
          <div className="page">
            <div className="card empty" role="alert">
              <h1 className="display display--small">The data could not be loaded</h1>
              <p>{error}</p>
              <p className="muted">
                Locally, run <code>python export_site.py</code> from the repository root, then reload.
              </p>
            </div>
          </div>
        )}
        {!data && !error && (
          <div className="page" aria-busy="true">
            <p className="muted">Loading forecasts…</p>
          </div>
        )}
        {data && route.page === "matches" && <MatchesPage data={data} param={route.param} />}
        {data && route.page === "record" && <RecordPage data={data} param={route.param} />}
        {data && route.page === "teams" && <TeamsPage data={data} param={route.param} />}
        {data && route.page === "method" && <MethodPage data={data} />}
      </main>

      {data && (
        <footer className="site-footer">
          <p>
            Data built {utcStamp(data.meta.generatedAt)} · model {data.meta.model.version} ·{" "}
            <a href={data.meta.repo}>Code on GitHub</a> · Data: Football-Data.co.uk
          </p>
        </footer>
      )}
    </>
  );
}
