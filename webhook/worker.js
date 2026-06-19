// Strava -> GitHub Actions relay (Cloudflare Worker)
// --------------------------------------------------
// This endpoint does three things:
//   1. GET  -> answer Strava's one-time subscription validation handshake.
//   2. POST -> on a real activity event, (re)start a debounce timer.
//   3. After the timer goes quiet -> trigger the GitHub workflow ONCE.
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

// The debounce timer. One shared instance handles all events.
export class StravaDebouncer extends DurableObject {
  // Called on every incoming Strava event. (Re)set the alarm to DEBOUNCE_MS from
  // now. A Durable Object has at most one alarm, so each call just pushes the
  // single pending wake-up further into the future — that's the whole debounce:
  // as long as events keep arriving, the alarm keeps sliding forward.
  async bump() {
    await this.ctx.storage.setAlarm(Date.now() + DEBOUNCE_MS);
  }

  // Fires once, DEBOUNCE_MS after the most recent bump() (i.e. once the edits
  // have gone quiet). This is where we actually trigger the GitHub workflow.
  async alarm() {
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
        // workflow_dispatch REQUIRES a `ref` (which branch to run on).
        body: JSON.stringify({ ref: REF }),
      }
    );

    // GitHub returns 204 on success. Log anything else so `wrangler tail` shows it.
    if (ghResponse.status !== 204) {
      console.log("GitHub dispatch failed:", ghResponse.status, await ghResponse.text());
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
    // We don't need to read Strava's payload; any event means "data changed."
    // Route it to the one shared Durable Object (fixed name "strava") so every
    // event in a burst shares the same timer, then reply "ok" fast.
    if (request.method === "POST") {
      const id = env.DEBOUNCER.idFromName("strava");
      const stub = env.DEBOUNCER.get(id);
      await stub.bump();

      // Always 200 back to Strava so it considers the event delivered.
      return new Response("ok", { status: 200 });
    }

    return new Response("Method not allowed", { status: 405 });
  },
};
