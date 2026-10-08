import { useEffect, useState } from "react";

// Hash routes (#/matches/5, #/teams/ARS) work on any static host, GitHub Pages included.
export type Page = "matches" | "record" | "teams" | "method";

export interface Route {
  page: Page;
  param: string | null;
}

const PAGES: Page[] = ["matches", "record", "teams", "method"];

export function parseHash(hash: string): Route {
  const [page, param] = hash.replace(/^#\/?/, "").split("/");
  const known = PAGES.find((p) => p === page);
  return { page: known ?? "matches", param: known && param ? decodeURIComponent(param) : null };
}

export function href(page: Page, param?: string | number): string {
  return param === undefined ? `#/${page}` : `#/${page}/${encodeURIComponent(String(param))}`;
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const update = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return route;
}

/** Navigate within the site; the back button then steps through matchweeks and clubs. */
export function go(page: Page, param?: string | number): void {
  window.location.hash = href(page, param);
}
