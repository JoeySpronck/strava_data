// Strava -> GitHub Actions relay (Cloudflare Worker)
// --------------------------------------------------
// ⚠️  AFTER EDITING THIS FILE YOU MUST RUN `wrangler deploy` (from webhook/).
//     Committing to git does NOT update the live Worker — Cloudflare only changes
//     on deploy. Skip it and production keeps running the OLD code while git looks
//     up to date (this exact drift once silently disabled the debounce below).
//     Verify after deploying: `wrangler tail`, edit an activity, confirm behavior.
// --------------------------------------------------
// This endpoint does three things:
//   1. GET  -> answer Strava's one-time subscription validation handshake.
//   2. POST -> on a real activity event, remember its activity id and (re)start a
//      debounce timer.
//   3. After the timer goes quiet -> trigger the GitHub workflow ONCE, passing every
//      id seen in the last 24 hours as the `refresh_ids` input so the workflow refetches
//      those activities' descriptions / private notes / streams instead of trusting its
//      cache. (Why 24 h and not just the new ids: see ID_MEMORY_MS below.)
//
// Why the debounce? A single user action often produces a burst of events:
//   - ActivityFix auto-edits the activity (1-2 edits),
//   - then you flip the activity from private to followers (another edit).
// Firing the workflow on every event would run the plot job 3x for one ride.
// Instead, each event pushes the timer 5 minutes into the future; the workflow
// only runs once 5 minutes pass with no new events (i.e. the edits have settled).
//
// A plain Worker is stateless (each request is a fresh invocation with no
// memory), so the timer lives in a Durable Object, which CAN hold state and
// schedule a wake-up via the Alarms API. We route every event to one shared DO
// instance, so all of a burst's events share the same timer.
//
// `env` holds the secrets we set with `wrangler secret put` (never in code):
//   env.STRAVA_VERIFY_TOKEN  - random string we invented; proves a GET is ours
//   env.GH_TOKEN             - fine-grained PAT scoped to strava_data, Actions:write only
// `env.DEBOUNCER` is the Durable Object binding declared in wrangler.toml.

import { DurableObject } from "cloudflare:workers";

const WORKFLOW_FILE = "update_plots.yml"; // the workflow we want to run
const REF = "main"; // which branch to run it on
const DEBOUNCE_MS = 3 * 60 * 1000; // wait this long after the LAST event before firing
// Every dispatch re-sends all ids seen this long ago or later. A run can take ~90 min
// (it waits out Strava's rate limit), and GitHub keeps only ONE queued run per
// concurrency group: a newer dispatch cancels the older queued one, ids and all.
// Re-sending recent ids means the run that does go ahead still refetches them. Cost:
// a handful of extra Strava requests per run.
const ID_MEMORY_MS = 24 * 60 * 60 * 1000;

// The debounce timer. One shared instance handles all events.
export class StravaDebouncer extends DurableObject {
  // Called on every incoming Strava event. (Re)set the alarm to DEBOUNCE_MS from
  // now. A Durable Object has at most one alarm, so each call just pushes the
  // single pending wake-up further into the future — that's the whole debounce:
  // as long as events keep arriving, the alarm keeps sliding forward.
  //
  // `activityId` (if any) is stored as {id: last seen, ms}, so a burst touching several
  // activities still ends in one dispatch that carries all of their ids.
  async bump(activityId) {
    if (activityId) {
      const seen = (await this.ctx.storage.get("seen")) || {};
      seen[activityId] = Date.now();
      await this.ctx.storage.put("seen", seen);
    }
    await this.ctx.storage.setAlarm(Date.now() + DEBOUNCE_MS);
  }

  // Fires once, DEBOUNCE_MS after the most recent bump() (i.e. once the edits
  // have gone quiet). This is where we actually trigger the GitHub workflow.
  async alarm() {
    // Forget ids older than ID_MEMORY_MS; send all the others (see ID_MEMORY_MS).
    const cutoff = Date.now() - ID_MEMORY_MS;
    const seen = (await this.ctx.storage.get("seen")) || {};
    const recent = Object.fromEntries(Object.entries(seen).filter(([, t]) => t >= cutoff));
    await this.ctx.storage.put("seen", recent);
    const ids = Object.keys(recent);
    const ghResponse = await fetch(
      // workflow_dispatch endpoint: targets ONE named workflow file.
      // Requires only the Actions:write permission (cannot modify code).
      `https://api.github.com/repos/${this.env.GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.env.GH_TOKEN}`,
          Accept: "application/vnd.github+json",
          "User-Agent": "strava-webhook-relay", // GitHub rejects requests without this
          "Content-Type": "application/json",
        },
        // workflow_dispatch REQUIRES a `ref` (which branch to run on). `refresh_ids`
        // must be declared as an input in update_plots.yml, or GitHub rejects it (422).
        body: JSON.stringify({ ref: REF, inputs: { refresh_ids: ids.join(",") } }),
      }
    );

    // GitHub returns 204 on success. Log anything else so `wrangler tail` shows it.
    // Ids are never removed on send, only by age, so a failed dispatch loses nothing:
    // the next event's dispatch still carries them.
    if (ghResponse.status !== 204) {
      console.log("GitHub dispatch failed:", ghResponse.status, await ghResponse.text());
    } else {
      console.log("Dispatched with refresh_ids:", ids.join(",") || "(none)");
    }
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // --- 1. Validation handshake (Strava sends this once, when we subscribe) ---
    // Strava calls: GET /?hub.mode=subscribe&hub.verify_token=XXX&hub.challenge=YYY
    // We confirm the token is ours, then echo the challenge back as JSON.
    if (request.method === "GET") {
      const mode = url.searchParams.get("hub.mode");
      const token = url.searchParams.get("hub.verify_token");
      const challenge = url.searchParams.get("hub.challenge");

      if (mode === "subscribe" && token === env.STRAVA_VERIFY_TOKEN) {
        // Strava requires exactly this shape, echoed within ~2 seconds.
        return Response.json({ "hub.challenge": challenge });
      }
      // Wrong/missing token -> reject. Keeps random internet scanners out.
      return new Response("Forbidden", { status: 403 });
    }

    // --- 2. Real activity event -> (re)start the debounce timer ---
    // Any event means "data changed". For activity events we also pass the activity id
    // (Strava's `object_id`) so the workflow refetches that activity's details. A body
    // we can't parse still restarts the timer: a plain refresh beats a missed one.
    // Route it to the one shared Durable Object (fixed name "strava") so every
    // event in a burst shares the same timer, then reply "ok" fast.
    if (request.method === "POST") {
      let activityId = null;
      try {
        const event = await request.json();
        if (event.object_type === "activity" && event.object_id) {
          activityId = String(event.object_id);
        }
      } catch (e) {
        console.log("Unparseable webhook body:", e);
      }
      const id = env.DEBOUNCER.idFromName("strava");
      const stub = env.DEBOUNCER.get(id);
      await stub.bump(activityId);

      // Always 200 back to Strava so it considers the event delivered.
      return new Response("ok", { status: 200 });
    }

    return new Response("Method not allowed", { status: 405 });
  },
};
