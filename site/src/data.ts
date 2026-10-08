import type { Match, Meta, Record_, SiteData, Team } from "./types";

async function load<T>(name: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`, {
    cache: "no-cache",
  });
  if (!response.ok) throw new Error(`data/${name}.json: HTTP ${response.status}`);
  return (await response.json()) as T;
}

/** Everything export_site.py wrote, fetched in parallel. */
export async function loadSiteData(): Promise<SiteData> {
  const [meta, matches, record, teams] = await Promise.all([
    load<Meta>("meta"),
    load<Match[]>("matches"),
    load<Record_>("record"),
    load<Team[]>("teams"),
  ]);
  return { meta, matches, record, teams };
}
