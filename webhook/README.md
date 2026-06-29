# Strava → GitHub Actions webhook (optional)

> **This whole folder is optional.** The repo works perfectly fine without it.
>
> Without the webhook, the plots refresh on the schedule already defined in
> [`.github/workflows/update_plots.yml`](../.github/workflows/update_plots.yml)
> (a daily `cron`) and whenever you push to `main`. The webhook just adds a
> *third* trigger: **plots update a few minutes after you finish or edit a
> Strava activity**, instead of waiting for the daily run.
>
> The Worker **debounces** events: one ride often fires a burst (ActivityFix
> auto-edits, then you flip private→followers). Rather than run the job once per
> event, each event restarts a 3-minute timer and the workflow runs **once**,
> 3 minutes after the edits go quiet. See [How the debounce works](#how-the-debounce-works).
>
> If "updated once a day" is good enough for you, **you can ignore this folder
> entirely.** Delete it and nothing breaks.

---

## What it does (the big picture)

A Strava webhook can't talk to GitHub directly — they're separate systems that
don't know about each other. So we put a tiny piece of glue in the middle: a
**Cloudflare Worker** (a ~50-line script that runs on Cloudflare's servers and
only wakes up when an HTTP request hits it).

```
You finish/edit a Strava activity
        │
        ▼
Strava's servers ──HTTP POST──► Cloudflare Worker (this folder)
                                     │  calls GitHub's API
                                     ▼
                          GitHub runs update_plots.yml
                                     │
                                     ▼
                     pulls Strava data → makes plots → pushes
```

The Worker does exactly two things:

1. **GET request** → answers Strava's one-time "are you really my endpoint?"
   validation handshake.
2. **POST request** → on a real activity event, (re)starts a 3-minute debounce
   timer. When the timer finally fires (no new events for 3 minutes), it calls
   GitHub's `workflow_dispatch` API — the same as pressing the **"Run workflow"**
   button in the Actions tab.

---

## How the debounce works

A plain Worker is **stateless**: every request is a fresh invocation that
remembers nothing about the previous one. So "wait 3 minutes, and if more edits
arrive, reset the wait" needs somewhere to keep the timer between requests.

That's a **Durable Object** (`StravaDebouncer` in `worker.js`): a tiny stateful
object that can schedule a wake-up via the [Alarms API](https://developers.cloudflare.com/durable-objects/api/alarms/).
Every Strava event is routed to **one shared instance** (named `"strava"`), which
sets its single alarm to "now + 3 minutes". Because a Durable Object has at most
one alarm, each new event just slides that one wake-up further into the future:

```
POST edit 1 (ActivityFix)   ──►  alarm = now + 3 min
POST edit 2 (ActivityFix)   ──►  alarm = now + 3 min   (reset)
POST edit 3 (private→followers) ─►  alarm = now + 3 min   (reset)
        … 3 minutes of no events …
   alarm() fires             ──►  ONE workflow_dispatch → plots regenerate
```

Change the wait by editing `DEBOUNCE_MS` near the top of `worker.js`, then
`wrangler deploy`.

> **Free tier.** The Durable Object uses the **SQLite** storage backend
> (declared via `new_sqlite_classes` in `wrangler.toml`), which is the kind
> available on the Workers Free plan — so this costs nothing.

---

## Files in this folder

| File | What it is |
|------|------------|
| `worker.js` | The Worker code. Handles the GET handshake + POST trigger. |
| `wrangler.toml` | Config for Cloudflare's CLI. The **only** thing a fork edits is `GITHUB_REPO`. No secrets in here. |
| `.gitignore` | Ignores local wrangler/Node junk (`.wrangler/`, `node_modules/`, `.dev.vars`). |
| `README.md` | This file. |

**Secrets are never in these files.** They live encrypted in Cloudflare and are
set with `wrangler secret put` (see Step 4).

---

## Setup from scratch

You need three accounts/things, all free:

- A **Cloudflare** account (hosts the Worker).
- **Node.js + npm** installed locally (only to install the deploy tool).
- Your **Strava API application** credentials (`client_id` + `client_secret`),
  from <https://www.strava.com/settings/api>. This repo stores them in
  `secrets/client_secrets.txt` as `client_id,client_secret`.

### Step 1 — Cloudflare account + the `wrangler` CLI

1. Sign up (no credit card needed for the free tier): <https://dash.cloudflare.com/sign-up>
2. Install `wrangler`, Cloudflare's deploy tool:
   ```bash
   npm install -g wrangler
   wrangler --version
   ```
3. Log in (opens a browser to authorize the CLI):
   ```bash
   wrangler login
   ```

### Step 2 — Create a scoped GitHub token

The Worker needs permission to trigger your workflow. We use a **fine-grained
Personal Access Token (PAT)** locked to *one repo* and *one permission*, so if it
ever leaked the worst case is "someone can re-run my plot job" — it can't touch
your code or other repos.

Create it at <https://github.com/settings/personal-access-tokens/new>:

- **Token name:** `strava-webhook-relay`
- **Expiration:** *No expiration* (the token is so narrowly scoped that the
  maintenance of rotating it isn't worth it — your call).
- **Resource owner:** your account.
- **Repository access:** *Only select repositories* → pick **this repo only**.
- **Permissions → Repository permissions → Actions:** *Read and write*.
  Everything else: *No access*.
  > Why `Actions` and not `Contents`? We trigger via the `workflow_dispatch` API,
  > which only needs `Actions: write` — it can run workflows but **cannot modify
  > files**. (The alternative, `repository_dispatch`, needs `Contents: write`,
  > which *can* push code → a much bigger risk. We deliberately avoid it.)
- Click **Generate token** and **copy it now** (GitHub shows it once). It starts
  with `github_pat_...`. Don't paste it into any file — you'll feed it to
  Cloudflare in Step 4.

### Step 3 — Point the Worker at your repo

Edit `wrangler.toml` and set your repo:

```toml
[vars]
GITHUB_REPO = "your-username/your-repo"
```

`WORKFLOW_FILE` and `REF` (the branch) are constants near the top of `worker.js`
— change them only if your workflow file isn't `update_plots.yml` or your branch
isn't `main`.

### Step 4 — Deploy, then add the two secrets

`wrangler secret put` attaches secrets to an **existing** Worker, so deploy first:

```bash
cd webhook
wrangler deploy
```

The first deploy asks you to register a `workers.dev` subdomain — say yes and
pick a name. Your public URL becomes:

```
https://strava-webhook-relay.<your-subdomain>.workers.dev
```

Now add the two secrets:

```bash
# 1. The GitHub token from Step 2 — paste it at the prompt (not echoed):
wrangler secret put GH_TOKEN

# 2. A random "verify token" you invent. Generate one and copy it:
openssl rand -hex 16
# ...then paste that value at the prompt:
wrangler secret put STRAVA_VERIFY_TOKEN
```

**Save the verify token** — you need the *exact same* value in Step 6. Confirm
both secrets exist (names only, values stay hidden):

```bash
wrangler secret list
```

### Step 5 — Test the handshake (before involving Strava)

> ⚠️ **Some corporate networks/VPNs block `*.workers.dev`.** If a request to your
> Worker resolves to a private `10.x.x.x` address or shows a certificate that
> isn't Cloudflare's, you're being filtered — test **off the VPN** (or from a
> phone on cellular). This does **not** affect production, because Strava calls
> your Worker from the public internet.

Wrong token should be rejected (`403 Forbidden`):

```bash
curl -s -w "\n[HTTP %{http_code}]\n" \
  "https://strava-webhook-relay.<your-subdomain>.workers.dev/?hub.mode=subscribe&hub.verify_token=WRONG&hub.challenge=test123"
```

Correct token should echo the challenge back (`{"hub.challenge":"test123"}`):

```bash
curl -s -w "\n[HTTP %{http_code}]\n" \
  "https://strava-webhook-relay.<your-subdomain>.workers.dev/?hub.mode=subscribe&hub.verify_token=YOUR_VERIFY_TOKEN&hub.challenge=test123"
```

### Step 6 — Register the subscription with Strava

This is the one-time POST that tells Strava where to send events. When Strava
receives it, it immediately fires the GET handshake at your Worker (which you
just tested), then replies with a subscription id.

Run it off-VPN. It reads your Strava creds from the file so you don't paste the
secret. Replace `YOUR_VERIFY_TOKEN` and the URL:

```bash
IFS=',' read CID CSECRET < ../secrets/client_secrets.txt
curl -s -X POST https://www.strava.com/api/v3/push_subscriptions \
  -F client_id="$CID" \
  -F client_secret="$CSECRET" \
  -F callback_url="https://strava-webhook-relay.<your-subdomain>.workers.dev/" \
  -F verify_token="YOUR_VERIFY_TOKEN"
```

- **Success:** `{"id": 1234567}` — you're subscribed. 🎉
- **`"code":"not verifiable"`** → handshake failed (usually a verify-token typo,
  or the Worker URL is unreachable from Strava).
- **`"already exists"`** → Strava allows only **one** subscription per app. List
  and delete the old one, then re-register:
  ```bash
  # list:
  curl -s -G https://www.strava.com/api/v3/push_subscriptions \
    -d client_id="$CID" -d client_secret="$CSECRET"
  # delete:
  curl -s -X DELETE "https://www.strava.com/api/v3/push_subscriptions/<id>?client_id=$CID&client_secret=$CSECRET"
  ```

### Step 7 — Test end to end

```bash
# Watch the Worker's live logs in one terminal:
wrangler tail
```

Then edit any Strava activity (change a title). You'll see `bump()` logged in
`wrangler tail` immediately, and **~3 minutes after your last edit** a new
**"Update plots from Strava"** run appears in your repo's **Actions** tab,
triggered by `workflow_dispatch`. Done — the loop runs on its own now.

You can also trigger it manually to test the Worker→GitHub link without Strava
(note the 3-minute debounce wait before the run appears):

```bash
curl -s -X POST https://strava-webhook-relay.<your-subdomain>.workers.dev/
# -> "ok" immediately; the workflow run appears ~3 min later in the Actions tab
```

---

## Common tasks

**Rotate the verify token** (e.g. if it leaked):

```bash
openssl rand -hex 16                      # make a new one, copy it
wrangler secret put STRAVA_VERIFY_TOKEN   # paste it
# then re-run Step 6 with the new value (Strava must know the new token too)
```

**Rotate the GitHub token:** regenerate it on GitHub, then
`wrangler secret put GH_TOKEN` again. Nothing else changes.

**Update the Worker code:** edit `worker.js`, then **`wrangler deploy`** (run from
`webhook/`). ⚠️ Mandatory and easy to forget: committing to git does **not** update
the live Worker — Cloudflare only changes on deploy. Skip it and production silently
keeps running the old code while the repo looks up to date. After deploying, verify
with `wrangler tail` + a test edit before trusting the change.

**Remove the webhook entirely:** delete the Strava subscription (Step 6's DELETE
command), delete the Worker in the Cloudflare dashboard, and delete this folder.
The repo falls back to its scheduled/push triggers.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| curl to the Worker hits a `10.x.x.x` IP / wrong cert | Corporate VPN/filter blocking `workers.dev` — test off-VPN. |
| `not verifiable` when registering | Verify token mismatch between Cloudflare and the Strava POST, or Worker URL wrong. |
| Handshake works but workflow never runs | Check `wrangler tail` for a `GitHub dispatch failed` log — usually the PAT lacks `Actions: write` or points at the wrong repo. |
| `already exists` on register | One subscription per Strava app — delete the old one first. |
| Workflow runs but fails | Not a webhook problem — the trigger worked. Debug the workflow/script separately. |

## Useful links

- Strava webhook docs: <https://developers.strava.com/docs/webhooks/>
- Strava API settings (your app): <https://www.strava.com/settings/api>
- Cloudflare Workers docs: <https://developers.cloudflare.com/workers/>
- Wrangler CLI docs: <https://developers.cloudflare.com/workers/wrangler/>
- GitHub fine-grained PATs: <https://github.com/settings/personal-access-tokens>
- GitHub `workflow_dispatch` API: <https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event>
