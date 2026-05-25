/**
 * Multi-Claude Extension
 *
 * Stores multiple Anthropic OAuth tokens in auth.json as "anthropic-N".
 * Switching copies the chosen token to the "anthropic" key.
 */

import { loginAnthropic } from "@earendil-works/pi-ai/oauth";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const ST = path.join(os.homedir(), ".pi", "agent", "multi-claude-state.json");
const PF = "anthropic-";
const ACT = "anthropic";
const TO = 10000;
const STALE = 5 * 60 * 1000;
const API = "https://api.anthropic.com";

interface St { email?: string; active?: boolean; usage?: { at: number; w: Record<string, { pct: number; reset?: number }> }; }
type Store = Record<string, St>;

const store = {
	load(): Store { try { return fs.existsSync(ST) ? JSON.parse(fs.readFileSync(ST, "utf-8")) : {}; } catch { return {}; } },
	save(s: Store) { fs.mkdirSync(path.dirname(ST), { recursive: true }); fs.writeFileSync(ST, JSON.stringify(s)); },
};

async function usage(ak: string) {
	try {
		const r = await fetch(`${API}/api/oauth/usage`, { headers: { Authorization: `Bearer ${ak}`, "anthropic-beta": "oauth-2025-04-20" }, signal: AbortSignal.timeout(TO) });
		if (!r.ok) return null;
		const d = (await r.json()) as any;
		const w: Record<string, { pct: number; reset?: number }> = {};
		if (d.five_hour?.utilization !== undefined) w["5h"] = { pct: d.five_hour.utilization, reset: d.five_hour.resets_at ? Date.parse(d.five_hour.resets_at) : undefined };
		if (d.seven_day?.utilization !== undefined) w["week"] = { pct: d.seven_day.utilization, reset: d.seven_day.resets_at ? Date.parse(d.seven_day.resets_at) : undefined };
		const mw = d.seven_day_sonnet || d.seven_day_opus;
		if (mw?.utilization !== undefined) w[d.seven_day_sonnet ? "sonnet" : "opus"] = { pct: mw.utilization };
		return { w };
	} catch { return null; }
}

async function profile(ak: string) {
	try {
		const r = await fetch(`${API}/api/oauth/profile`, { headers: { Authorization: `Bearer ${ak}` }, signal: AbortSignal.timeout(TO) });
		return r.ok ? ((await r.json()) as any)?.account?.email as string | undefined : undefined;
	} catch { return undefined; }
}

function pn(n: number) { return `${PF}${n}`; }

// ---------------------------------------------------------------------------
// TUI list
// ---------------------------------------------------------------------------

interface Row {
	i: number; email: string; win: Array<{ n: string; pct: number; reset?: number; clr: string }>; err?: string; active: boolean;
}

class List {
	private rs: Row[] = [];
	private loading = true;
	private sel = 0;
	private tu: { requestRender: () => void };
	private th: any;
	private done: () => void;
	private ctx: any;
	private busy = "";

	constructor(tu: any, th: any, done: () => void, ctx: any) {
		this.tu = tu; this.th = th; this.done = done; this.ctx = ctx; void this.init();
	}

	private d(s: string) { return this.th.fg("muted", s); }
	private b(s: string) { return this.th.bold(s); }
	private a(s: string) { return this.th.fg("accent", s); }

	private async init() {
		const s = store.load();
		const rs: Row[] = [];
		for (const [k, st] of Object.entries(s).sort()) {
			if (!k.startsWith(PF)) continue;
			const i = parseInt(k.slice(PF.length), 10);
			const ak = await this.ctx.modelRegistry.authStorage.getApiKey(k);
			if (!ak) { delete s[k]; continue; }

			let w = st.usage?.w;
			if (!st.usage || Date.now() - st.usage.at > STALE) {
				const u = await usage(ak);
				if (u) { w = u.w; s[k].usage = { at: Date.now(), w: u.w }; }
			}

			const wins: Row["win"] = [];
			if (w) for (const [wn, wd] of Object.entries(w).sort((a, b) => ({ week: 0, "5h": 1, sonnet: 2, opus: 3 } as any)[a[0]] - ({ week: 0, "5h": 1, sonnet: 2, opus: 3 } as any)[b[0]])) {
				const rem = 100 - wd.pct;
				wins.push({ n: wn, pct: wd.pct, reset: wd.reset, clr: rem <= 10 ? "error" : rem <= 30 ? "warning" : "success" });
			}
			rs.push({ i, email: st.email ?? "unknown", win: wins, err: w ? undefined : "fetch failed", active: !!st.active });
		}
		store.save(s);
		this.rs = rs;
		const ai = rs.findIndex(r => r.active);
		if (ai >= 0) this.sel = ai;
		this.loading = false; this.tu.requestRender();
	}

	handleInput(ev: string): void {
		if (this.busy) return;
		if (matchesKey(ev, "escape")) { this.done(); return; }
		if (matchesKey(ev, "up") || ev === "k") { this.sel = Math.max(0, this.sel - 1); this.tu.requestRender(); return; }
		if (matchesKey(ev, "down") || ev === "j") { this.sel = Math.min(this.rs.length - 1, this.sel + 1); this.tu.requestRender(); return; }
		if (matchesKey(ev, "enter")) { void this.withBusy("switch", () => this.doSwitch()); return; }
		if (matchesKey(ev, "backspace") || matchesKey(ev, "delete")) { void this.withBusy("remove", () => this.doRemove()); }
	}

	private async withBusy(label: string, fn: () => Promise<void>) {
		this.busy = label; this.tu.requestRender();
		try { await fn(); }
		catch { this.busy = ""; this.tu.requestRender(); }
	}

	private async doSwitch() {
		const r = this.rs[this.sel]; if (!r) return;
		const s = store.load();
		// Clear previous active, set new
		for (const k of Object.keys(s)) if (k.startsWith(PF)) s[k].active = false;
		s[pn(r.i)] = { ...s[pn(r.i)], active: true };
		store.save(s);
		const cred = this.ctx.modelRegistry.authStorage.get(pn(r.i));
		if (cred) this.ctx.modelRegistry.authStorage.set(ACT, cred);
		this.done();
	}

	private async doRemove() {
		const r = this.rs[this.sel]; if (!r) return;
		const s = store.load();
		delete s[pn(r.i)];
		if (r.active) this.ctx.modelRegistry.authStorage.remove(ACT);
		this.ctx.modelRegistry.authStorage.remove(pn(r.i));
		store.save(s);
		this.busy = ""; this.loading = true; void this.init().then(() => { this.sel = Math.min(this.sel, this.rs.length - 1); this.tu.requestRender(); });
	}

	invalidate(): void {}
	dispose(): void {}

	render(w: number): string[] {
		const t = this.th, iw = w - 4, hl = "─".repeat(w - 2);
		const bx = (c: string) => this.d("│ ") + c + " ".repeat(Math.max(0, iw - c.replace(/\x1b\[[0-9;]*m/g, "").length)) + this.d(" │");
		const l: string[] = [];

		if (this.busy) {
			const act = this.busy === "switch" ? "switching" : "removing";
			l.push(this.d(`╭${hl}╮`), bx(this.b(this.a("multi-claude"))), this.d(`├${hl}┤`));
			l.push(bx(`${act} [${this.rs[this.sel]?.i}] ${this.rs[this.sel]?.email}...`));
			l.push(this.d(`╰${hl}╯`)); return l;
		}

		l.push(this.d(`╭${hl}╮`), bx(this.b(this.a("multi-claude"))), this.d(`├${hl}┤`));

		if (this.loading) l.push(bx("loading..."));
		else if (!this.rs.length) { l.push(bx("no accounts"), bx(""), bx(this.d("/multi-claude add"))); }
		else for (let i = 0; i < this.rs.length; i++) {
			const r = this.rs[i];
			l.push(bx(`${i === this.sel ? t.fg("accent", "▸ ") : "  "}${this.b(`[${r.i}]`)} ${r.email}${r.active ? t.fg("success", " ●") : ""}`));
			if (r.err) { l.push(bx(this.d(`   ${r.err}`))); continue; }
			for (const w of r.win) {
				const f = Math.min(10, Math.round(w.pct / 10)), e = 10 - f;
				const bar = t.fg(w.clr, "█".repeat(f)) + this.d("░".repeat(e));
				const rs = w.reset ? this.d(` ${fmt(new Date(w.reset))}`) : "";
				l.push(bx(`   ${w.n.padEnd(6)} ${bar} ${w.pct.toFixed(0).padStart(3)}%${rs}`));
			}
		}

		l.push(this.d(`├${hl}┤`), bx(this.d("↑↓ select  ↵ switch  ⌫ remove  esc close")), this.d(`╰${hl}╯`));
		return l;
	}
}

function fmt(d: Date): string {
	const df = d.getTime() - Date.now(); if (df < 0) return "now";
	const m = Math.floor(df / 60000); if (m < 60) return `in ${m}m`;
	const h = Math.floor(m / 60); if (h < 24) return `in ${h}h${m % 60 ? ` ${m % 60}m` : ""}`;
	return `in ${Math.floor(h / 24)}d`;
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	pi.registerCommand("multi-claude", {
		description: "Manage multiple Anthropic Claude accounts",
		getArgumentCompletions(pref: string) {
			const p = pref.trimStart().split(/\s+/).filter(Boolean);
			if (!p.length) return [{ value: "list", label: "list" }, { value: "add", label: "add" }];
			if (p.length === 1 && !pref.endsWith(" "))
				return ["list", "add"].filter(s => s.startsWith(p[0])).map(s => ({ value: s, label: s }));
			return null;
		},
		handler: async (args, ctx) => {
			const p = (args || "").trim().split(/\s+/).filter(Boolean);
			const sub = p[0]?.toLowerCase();

			if (sub === "add") {
				try {
					const creds = await loginAnthropic({
						onAuth: ({ url }) => {
							ctx.ui.notify(`Open: ${url}`, "info");
							void import("node:child_process").then(({ exec }) => {
								exec(process.platform === "darwin" ? `open '${url}'` : process.platform === "win32" ? `start "" "${url}"` : `xdg-open '${url}'`);
							});
						},
						onPrompt: async () => {
							const v = await ctx.ui.input("Paste the Anthropic authorization code:");
							if (!v?.trim()) throw new Error("Cancelled");
							return v.trim();
						},
					});

					// Find next index
					let idx = 0;
					for (const k of ctx.modelRegistry.authStorage.list()) {
						if (k.startsWith(PF)) { const n = parseInt(k.slice(PF.length), 10); if (!isNaN(n) && n >= idx) idx = n + 1; }
					}

					ctx.modelRegistry.authStorage.set(pn(idx), { type: "oauth", ...creds });
					const em = await profile(creds.access);
					const s = store.load();
					for (const k of Object.keys(s)) if (k.startsWith(PF)) s[k].active = false;
					s[pn(idx)] = { email: em, active: true };
					store.save(s);

					ctx.modelRegistry.authStorage.set(ACT, { type: "oauth", ...creds });
					ctx.ui.notify(`Added & switched to [${idx}]${em ? ` (${em})` : ""}`, "success");
				} catch (e: any) { ctx.ui.notify(`Failed: ${e?.message || e}`, "error"); }
				return;
			}

			if (sub === "list" || !sub) {
				await ctx.ui.custom((tu: any, th: any, _kb: any, done: () => void) => new List(tu, th, done, ctx));
				return;
			}

			ctx.ui.notify("/multi-claude add    Add account\n/multi-claude list   List (↑↓ ↵ switch, ⌫ remove)", "info");
		},
	});
}
