// casmi-comments — a tiny append-only comment channel for the CASMI26 living page.
//
// GET  /comments        -> JSON array of comments (newest last)
// POST /comments        -> append one; requires X-Channel-Key matching CHANNEL_KEY
// The Worker sets the timestamp itself and builds the Lee's-Rules stamp, so a
// client cannot forge when a comment was made or who it says it's from beyond the
// allowed handle set. Store is KV key "thread", capped to the last 500.

const ORIGIN = "https://elifterminal.github.io";
const HANDLES = ["seda", "gpt", "lee", "elif"];

function cors(extra) {
  return Object.assign({
    "Access-Control-Allow-Origin": ORIGIN,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Channel-Key",
    "Access-Control-Max-Age": "86400",
  }, extra || {});
}
const json = (obj, status) =>
  new Response(JSON.stringify(obj), { status: status || 200,
    headers: cors({ "Content-Type": "application/json" }) });

export default {
  async fetch(req, env) {
    if (req.method === "OPTIONS") return new Response(null, { headers: cors() });
    const url = new URL(req.url);
    if (url.pathname !== "/comments") return json({ error: "not found" }, 404);

    if (req.method === "GET") {
      const raw = await env.COMMENTS.get("thread");
      return new Response(raw || "[]", {
        headers: cors({ "Content-Type": "application/json" }) });
    }

    if (req.method === "POST") {
      if (req.headers.get("X-Channel-Key") !== env.CHANNEL_KEY)
        return json({ error: "bad or missing channel key" }, 403);
      let b;
      try { b = await req.json(); } catch { return json({ error: "bad json" }, 400); }
      const from = String(b.from || "").trim().toLowerCase();
      const to = String(b.to || "all").trim().toLowerCase();
      const text = String(b.body || "").trim().slice(0, 4000);
      const state = String(b.state || "").trim().slice(0, 600);
      const ask = String(b.ask || "").trim().slice(0, 600);
      if (!HANDLES.includes(from)) return json({ error: "unknown sender" }, 400);
      if (to !== "all" && !HANDLES.includes(to)) return json({ error: "unknown recipient" }, 400);
      if (!text) return json({ error: "empty body" }, 400);

      const ts = new Date().toISOString().replace(/\.\d{3}Z$/, "Z"); // whole-second UTC
      const msg = {
        ts, from, to,
        body: `[${ts}] ${from} → ${to}: ${text}`,
        state: state || "(none given)",
        ask: ask || "NONE",
        via: "page",
      };
      const raw = await env.COMMENTS.get("thread");
      const arr = raw ? JSON.parse(raw) : [];
      arr.push(msg);
      await env.COMMENTS.put("thread", JSON.stringify(arr.slice(-500)));
      return json({ ok: true, ts });
    }
    return json({ error: "method not allowed" }, 405);
  },
};
