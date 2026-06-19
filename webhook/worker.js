// Strava -> GitHub Actions relay (Cloudflare Worker)
// --------------------------------------------------
// This tiny endpoint does exactly two things:
//   1. GET  -> answer Strava's one-time subscription validation handshake.
//   2. POST -> on a real activity event, trigger the GitHub workflow.
//
// `env` holds the secrets we set with `wrangler secret put` (never in code):
//   env.STRAVA_VERIFY_TOKEN  - random string we invented; proves a GET is ours
//   env.GH_TOKEN             - fine-grained PAT scoped to strava_data, Actions:write only

const WORKFLOW_FILE = "update_plots.yml"; // the workflow we want to run
const REF = "main"; // which branch to run it on

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Configured once in wrangler.toml [vars] (e.g. "owner/repo"), not in code.
    const GITHUB_REPO = env.GITHUB_REPO;

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

    // --- 2. Real activity event -> trigger the GitHub workflow ---
    // We don't even need to read Strava's payload; any event means "data changed,
    // go regenerate the plots." We fire workflow_dispatch and reply "ok" fast.
    if (request.method === "POST") {
      const ghResponse = await fetch(
        // workflow_dispatch endpoint: targets ONE named workflow file.
        // Requires only the Actions:write permission (cannot modify code).
        `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${env.GH_TOKEN}`,
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

      // Always 200 back to Strava so it considers the event delivered.
      return new Response("ok", { status: 200 });
    }

    return new Response("Method not allowed", { status: 405 });
  },
};
