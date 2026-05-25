/**
 * Multi-Claude Extension
 *
 * Stores multiple Anthropic OAuth tokens in auth.json as "anthropic-N".
 * No state file, no cached metadata — everything fetched live.
 * Active account = whichever token is under the "anthropic" key.
 */

import { loginAnthropic } from "@earendil-works/pi-ai/oauth";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";

const PF = "anthropic-";
const ACT = "anthropic";
const TO = 10000;
const API = "https://api.anthropic.com";

async function fetchUsage(ak: string) {
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

async function fetchProfile(ak: string) {
	try {
		const r = await fetch(`${API}/api/oauth/profile`, { headers: { Authorization: `Bearer ${ak}` }, signal: AbortSignal.timeout(TO) });
		if (!r.ok) return undefined;
		const d = (await r.json()) as any;
		const acct = d.account, org = d.organization;
		let plan: string | undefined;
		if (acct?.has_claude_max) plan = "max";
		else if (acct?.has_claude_pro) plan = "pro";
		else if (org?.rate_limit_tier === "default_raven") plan = org?.seat_tier === "team_premium" ? "team premium" : "team standard";
		return { email: acct?.email as string | undefined, plan };
	} catch { return undefined; }
}

function pn(n: number) { return `${PF}${n}`; }

// ---------------------------------------------------------------------------
// Auth helpers — everything in auth.json
// ---------------------------------------------------------------------------

function getAccounts(as: any): string[] {
	return (as.list?.() ?? []).filter((k: string) => k.startsWith(PF)).sort();
}

function getActive(as: any): string | undefined {
	const cred = as.get(ACT);
	if (!cred) return undefined;
	for (const k of getAccounts(as)) {
		const v = as.get(k);
		if (v && v.type === cred.type && v.access === cred.access) return k;
	}
	return undefined;
}

// ---------------------------------------------------------------------------
// TUI list
// ---------------------------------------------------------------------------

interface Row {
	key: string; i: number; email: string; plan?: string; win: Array<{ n: string; pct: number; reset?: number; clr: string }>; err?: string; active: boolean;
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
		const as = this.ctx.modelRegistry.authStorage;
		const accounts = getAccounts(as);
		const activeKey = getActive(as);
		const rs: Row[] = [];

		for (const k of accounts) {
			const i = parseInt(k.slice(PF.length), 10);
			const v = as.get(k);
			if (!v) continue;

			const [u, p] = await Promise.all([fetchUsage(v.access), fetchProfile(v.access)]);
			const wins: Row["win"] = [];
			if (u?.w) for (const [wn, wd] of Object.entries(u.w).sort((a, b) => ({ week: 0, "5h": 1, sonnet: 2, opus: 3 } as any)[a[0]] - ({ week: 0, "5h": 1, sonnet: 2, opus: 3 } as any)[b[0]])) {
				const rem = 100 - wd.pct;
				wins.push({ n: wn, pct: wd.pct, reset: wd.reset, clr: rem <= 10 ? "error" : rem <= 30 ? "warning" : "success" });
			}
			rs.push({ key: k, i, email: p?.email ?? "unknown", plan: p?.plan, win: wins, err: u ? undefined : "fetch failed", active: k === activeKey });
		}

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
		if (ev === "a") { void this.withBusy("add", () => this.doAdd()); return; }
		if (matchesKey(ev, "backspace") || matchesKey(ev, "delete")) { void this.withBusy("remove", () => this.doRemove()); }
	}

	private async withBusy(label: string, fn: () => Promise<void>) {
		this.busy = label; this.tu.requestRender();
		try { await fn(); } catch { this.busy = ""; this.tu.requestRender(); }
	}

	private async doSwitch() {
		const r = this.rs[this.sel]; if (!r) return;
		const as = this.ctx.modelRegistry.authStorage;
		const cred = as.get(r.key);
		if (cred) as.set(ACT, cred);
		this.done();
	}

	private async doAdd() {
		try {
			const creds = await loginAnthropic({
				onAuth: ({ url }) => {
					this.ctx.ui.notify(`Open: ${url}`, "info");
					void import("node:child_process").then(({ exec }) => {
						exec(process.platform === "darwin" ? `open '${url}'` : process.platform === "win32" ? `start "" "${url}"` : `xdg-open '${url}'`);
					});
				},
				onPrompt: async () => {
					const v = await this.ctx.ui.input("Paste the Anthropic authorization code:");
					if (!v?.trim()) throw new Error("Cancelled");
					return v.trim();
				},
			});

			const as = this.ctx.modelRegistry.authStorage;

			let idx = 0;
			for (const k of as.list()) {
				if (k.startsWith(PF)) { const n = parseInt(k.slice(PF.length), 10); if (!isNaN(n) && n >= idx) idx = n + 1; }
			}

			as.set(pn(idx), { type: "oauth", ...creds });
			as.set(ACT, { type: "oauth", ...creds });

			this.ctx.ui.notify(`Added & switched to [${idx}]`, "success");
		} catch (e: any) { this.ctx.ui.notify(`Failed: ${e?.message || e}`, "error"); }
		this.busy = ""; this.loading = true; void this.init().then(() => { this.tu.requestRender(); });
	}

	private async doRemove() {
		const r = this.rs[this.sel]; if (!r) return;
		const as = this.ctx.modelRegistry.authStorage;
		if (r.active) as.remove(ACT);
		as.remove(r.key);
		this.busy = ""; this.loading = true; void this.init().then(() => { this.sel = Math.min(this.sel, this.rs.length - 1); this.tu.requestRender(); });
	}

	invalidate(): void {}
	dispose(): void {}

	render(w: number): string[] {
		const t = this.th, iw = w - 4, hl = "─".repeat(w - 2);
		const bx = (c: string) => this.d("│ ") + c + " ".repeat(Math.max(0, iw - c.replace(/\x1b\[[0-9;]*m/g, "").length)) + this.d(" │");
		const l: string[] = [];

		if (this.busy) {
			l.push(this.d(`╭${hl}╮`), bx(this.b(this.a("multi-claude"))), this.d(`├${hl}┤`));
			l.push(bx(`${this.busy}...`));
			l.push(this.d(`╰${hl}╯`)); return l;
		}

		l.push(this.d(`╭${hl}╮`), bx(this.b(this.a("multi-claude"))), this.d(`├${hl}┤`));

		if (this.loading) l.push(bx("loading..."));
		else if (!this.rs.length) { l.push(bx("no accounts"), bx(""), bx(this.d("a  add account"))); }
		else for (let i = 0; i < this.rs.length; i++) {
			const r = this.rs[i];
			const planLabel = r.plan ? t.fg("accent", ` ${r.plan}`) : "";
			l.push(bx(`${i === this.sel ? t.fg("accent", "▸ ") : "  "}${this.b(`[${r.i}]`)} ${r.email}${planLabel}${r.active ? t.fg("success", " ●") : ""}`));
			if (r.err) { l.push(bx(this.d(`   ${r.err}`))); continue; }
			for (const w of r.win) {
				const f = Math.min(10, Math.round(w.pct / 10)), e = 10 - f;
				const bar = t.fg(w.clr, "█".repeat(f)) + this.d("░".repeat(e));
				const rs = w.reset ? this.d(` ${fmt(new Date(w.reset))}`) : "";
				l.push(bx(`   ${w.n.padEnd(6)} ${bar} ${w.pct.toFixed(0).padStart(3)}%${rs}`));
			}
		}

		l.push(this.d(`├${hl}┤`), bx(this.d("↑↓ select  a add  ↵ switch  ⌫ remove  esc close")), this.d(`╰${hl}╯`));
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
		handler: async (_args, ctx) => {
			await ctx.ui.custom((tu: any, th: any, _kb: any, done: () => void) => new List(tu, th, done, ctx));
		},
	});
}
