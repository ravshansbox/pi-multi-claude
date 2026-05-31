import { loginAnthropic, refreshAnthropicToken } from "@earendil-works/pi-ai/oauth";
import { type ExtensionAPI, type ExtensionCommandContext, type AuthStorage, type Theme, type OAuthCredential } from "@earendil-works/pi-coding-agent";
import { matchesKey, type TUI, type KeybindingsManager, type Component } from "@earendil-works/pi-tui";

const ACCOUNT_PREFIX = "anthropic-";
const ACTIVE_KEY = "anthropic";
const FETCH_TIMEOUT_MS = 10_000;
const ANTHROPIC_API_BASE = "https://api.anthropic.com";

const DEBUG = !!process.env.MULTI_CLAUDE_DEBUG;

function debug(...args: unknown[]): void {
	if (DEBUG) console.error("[multi-claude]", ...args);
}

interface AnthropicUsageWindow {
	utilization: number;
	resets_at?: string;
}

interface AnthropicUsageResponse {
	five_hour?: AnthropicUsageWindow;
	seven_day?: AnthropicUsageWindow;
}

interface AnthropicProfileAccount {
	email?: string;
	has_claude_max?: boolean;
	has_claude_pro?: boolean;
}

interface AnthropicProfileOrg {
	rate_limit_tier?: string;
	seat_tier?: string;
}

interface AnthropicProfileResponse {
	account?: AnthropicProfileAccount;
	organization?: AnthropicProfileOrg;
}

async function fetchUsage(accessToken: string) {
	try {
		const response = await fetch(`${ANTHROPIC_API_BASE}/api/oauth/usage`, {
			headers: {
				Authorization: `Bearer ${accessToken}`,
				"anthropic-beta": "oauth-2025-04-20",
			},
			signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
		});
		if (!response.ok) {
			debug("fetchUsage non-ok", response.status);
			return null;
		}

		const data: AnthropicUsageResponse = await response.json();
		const windows: Record<string, { percent: number; reset?: number }> = {};

		if (data.five_hour?.utilization !== undefined) {
			windows["5h"] = {
				percent: data.five_hour.utilization,
				reset: data.five_hour.resets_at ? Date.parse(data.five_hour.resets_at) : undefined,
			};
		}

		if (data.seven_day?.utilization !== undefined) {
			windows["week"] = {
				percent: data.seven_day.utilization,
				reset: data.seven_day.resets_at ? Date.parse(data.seven_day.resets_at) : undefined,
			};
		}

		return { windows };
	} catch (error) {
		debug("fetchUsage error", error);
		return null;
	}
}

const USAGE_SORT_ORDER: Record<string, number> = { week: 0, "5h": 1 };

async function fetchProfile(accessToken: string) {
	try {
		const response = await fetch(`${ANTHROPIC_API_BASE}/api/oauth/profile`, {
			headers: { Authorization: `Bearer ${accessToken}` },
			signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
		});
		if (!response.ok) return undefined;

		const data: AnthropicProfileResponse = await response.json();
		const account = data.account;
		const organization = data.organization;

		let plan: string | undefined;
		if (account?.has_claude_max) {
			plan = "max";
		} else if (account?.has_claude_pro) {
			plan = "pro";
		} else if (organization?.rate_limit_tier === "default_raven") {
			plan = organization?.seat_tier === "team_premium" ? "team premium" : "team standard";
		}

		return { email: account?.email, plan };
	} catch (error) {
		debug("fetchProfile error", error);
		return undefined;
	}
}

function makeAccountKey(index: number): string {
	return `${ACCOUNT_PREFIX}${index}`;
}

function getAccountKeys(authStorage: AuthStorage): string[] {
	return authStorage
		.list()
		.filter((key) => key.startsWith(ACCOUNT_PREFIX))
		.sort();
}

function getActiveAccountKey(authStorage: AuthStorage): string | undefined {
	const activeCredential = authStorage.get(ACTIVE_KEY);
	if (!activeCredential) return undefined;

	for (const accountKey of getAccountKeys(authStorage)) {
		const credential = authStorage.get(accountKey);
		if (
			credential &&
			credential.type === "oauth" &&
			(credential as OAuthCredential).access === (activeCredential as OAuthCredential).access
		) {
			return accountKey;
		}
	}
	return undefined;
}

type UsageColor = "error" | "warning" | "success";

interface UsageWindowRow {
	name: string;
	percent: number;
	reset?: number;
	color: UsageColor;
}

interface AccountRow {
	key: string;
	index: number;
	email: string;
	plan?: string;
	usageWindows: UsageWindowRow[];
	error?: string;
	active: boolean;
}

async function getAccessToken(authStorage: AuthStorage, key: string): Promise<string | undefined> {
	const credential = authStorage.get(key);
	if (!credential || credential.type !== "oauth") return undefined;
	if (Date.now() < credential.expires) return credential.access;

	// Never refresh the provider's current/active key directly. Refresh only the
	// numbered account copies, then copy one into ACTIVE_KEY if needed.
	if (key === ACTIVE_KEY) return credential.access;

	const freshCredentials = await refreshAnthropicToken(credential.refresh).catch((error: unknown) => {
		debug("refreshAnthropicToken error", key, error);
		return undefined;
	});
	if (!freshCredentials) return undefined;

	authStorage.set(key, { type: "oauth", ...freshCredentials });
	return freshCredentials.access;
}

class AccountList implements Component {
	private rows: AccountRow[] = [];
	private loading = true;
	private selectedIndex = 0;
	private tui: TUI;
	private theme: Theme;
	private done: (_?: unknown) => void;
	private context: ExtensionCommandContext;
	private busy = "";

	constructor(
		tui: TUI,
		theme: Theme,
		_keybindings: KeybindingsManager,
		done: (_?: unknown) => void,
		context: ExtensionCommandContext,
	) {
		this.tui = tui;
		this.theme = theme;
		this.done = done;
		this.context = context;
		void this.init();
	}

	private dim(text: string): string {
		return this.theme.fg("muted", text);
	}

	private bold(text: string): string {
		return this.theme.bold(text);
	}

	private accent(text: string): string {
		return this.theme.fg("accent", text);
	}

	private async init() {
		const authStorage = this.context.modelRegistry.authStorage;
		const accountKeys = getAccountKeys(authStorage);
		const activeKey = getActiveAccountKey(authStorage);
		const rows: AccountRow[] = [];

		for (const accountKey of accountKeys) {
			const accountIndex = parseInt(accountKey.slice(ACCOUNT_PREFIX.length), 10);
			const credential = authStorage.get(accountKey);
			if (!credential) continue;

			const accessToken = await getAccessToken(authStorage, accountKey);
			if (!accessToken) {
				rows.push({
					key: accountKey,
					index: accountIndex,
					email: "unknown",
					usageWindows: [],
					error: "auth expired",
					active: accountKey === activeKey,
				});
				continue;
			}

			const [usage, profile] = await Promise.all([
				fetchUsage(accessToken),
				fetchProfile(accessToken),
			]);

			const usageWindows: UsageWindowRow[] = [];
			if (usage?.windows) {
				const entries = Object.entries(usage.windows).sort(
					(a, b) => (USAGE_SORT_ORDER[a[0]] ?? 99) - (USAGE_SORT_ORDER[b[0]] ?? 99),
				);
				for (const [windowName, windowData] of entries) {
					const remaining = 100 - windowData.percent;
					usageWindows.push({
						name: windowName,
						percent: windowData.percent,
						reset: windowData.reset,
						color: remaining <= 10 ? "error" : remaining <= 30 ? "warning" : "success",
					});
				}
			}

			rows.push({
				key: accountKey,
				index: accountIndex,
				email: profile?.email ?? "unknown",
				plan: profile?.plan,
				usageWindows,
				error: usage ? undefined : "fetch failed",
				active: accountKey === activeKey,
			});
		}

		rows.sort(compareRowsByWeeklyReset);
		this.rows = rows;

		const activeIndex = rows.findIndex((row) => row.active);
		if (activeIndex >= 0) this.selectedIndex = activeIndex;

		this.loading = false;
		this.tui.requestRender();
	}

	handleInput(event: string): void {
		if (this.busy) return;

		if (matchesKey(event, "escape")) {
			this.done(undefined);
			return;
		}

		if (matchesKey(event, "up") || event === "k") {
			this.selectedIndex = Math.max(0, this.selectedIndex - 1);
			this.tui.requestRender();
			return;
		}

		if (matchesKey(event, "down") || event === "j") {
			this.selectedIndex = Math.min(this.rows.length - 1, this.selectedIndex + 1);
			this.tui.requestRender();
			return;
		}

		if (matchesKey(event, "enter")) {
			void this.withBusy("switch", () => this.switchAccount());
			return;
		}

		if (event === "a") {
			void this.withBusy("add", () => this.addAccount());
			return;
		}

		if (matchesKey(event, "backspace") || matchesKey(event, "delete")) {
			void this.withBusy("remove", () => this.removeAccount());
		}
	}

	private async withBusy(label: string, action: () => Promise<void>) {
		this.busy = label;
		this.tui.requestRender();
		try {
			await action();
		} finally {
			this.busy = "";
			this.tui.requestRender();
		}
	}

	private async switchAccount() {
		const row = this.rows[this.selectedIndex];
		if (!row || row.active) return;

		const authStorage = this.context.modelRegistry.authStorage;
		await getAccessToken(authStorage, row.key);
		const credential = authStorage.get(row.key);
		if (credential) {
			authStorage.set(ACTIVE_KEY, credential);
		}

		this.done(undefined);

		try {
			await this.context.reload();
		} catch (error) {
			this.context.ui.notify(
				`Reload failed: ${error instanceof Error ? error.message : String(error)}`,
				"error",
			);
		}
	}

	private async addAccount() {
		try {
			const credentials = await loginAnthropic({
				onAuth: ({ url, instructions }) => {
					this.context.ui.notify(`Open: ${url}`, "info");
					if (instructions) this.context.ui.notify(instructions, "info");

					void import("node:child_process").then(({ exec }) => {
						let openCmd: string;
						if (process.platform === "darwin") {
							openCmd = `open '${url}'`;
						} else if (process.platform === "win32") {
							openCmd = `start "" "${url}"`;
						} else {
							openCmd = `xdg-open '${url}'`;
						}
						exec(openCmd);
					});
				},
				onProgress: (message: string) => this.context.ui.notify(message, "info"),
				onPrompt: async ({ message }: { message: string }) => {
					const code = await this.context.ui.input(message);
					if (!code?.trim()) throw new Error("Cancelled");
					return code.trim();
				},
			});

			const authStorage = this.context.modelRegistry.authStorage;

			let nextIndex = 0;
			for (const key of authStorage.list()) {
				if (key.startsWith(ACCOUNT_PREFIX)) {
					const existingIndex = parseInt(key.slice(ACCOUNT_PREFIX.length), 10);
					if (!isNaN(existingIndex) && existingIndex >= nextIndex) {
						nextIndex = existingIndex + 1;
					}
				}
			}

			authStorage.set(makeAccountKey(nextIndex), { type: "oauth", ...credentials });
			authStorage.set(ACTIVE_KEY, { type: "oauth", ...credentials });

			this.context.ui.notify("Added & switched account", "info");
		} catch (error) {
			this.context.ui.notify(
				`Failed: ${error instanceof Error ? error.message : String(error)}`,
				"error",
			);
		}

		this.loading = true;
		void this.init().then(() => {
			this.tui.requestRender();
		});
	}

	private async removeAccount() {
		const row = this.rows[this.selectedIndex];
		if (!row) return;

		const authStorage = this.context.modelRegistry.authStorage;
		if (row.active) {
			authStorage.remove(ACTIVE_KEY);
		}
		authStorage.remove(row.key);

		this.loading = true;
		void this.init().then(() => {
			this.selectedIndex = Math.max(0, Math.min(this.selectedIndex, this.rows.length - 1));
			this.tui.requestRender();
		});
	}

	invalidate(): void {}
	dispose(): void {}

	render(width: number): string[] {
		const theme = this.theme;
		const innerWidth = width - 4;
		const horizontalLine = "─".repeat(width - 2);

		const boxLine = (content: string): string => {
			const visible = content.replace(/\x1b\[[0-9;]*m/g, "");
			const padding = " ".repeat(Math.max(0, innerWidth - visible.length));
			return this.dim("│ ") + content + padding + this.dim(" │");
		};

		const lines: string[] = [];

		if (this.busy) {
			lines.push(
				this.dim(`╭${horizontalLine}╮`),
				boxLine(this.bold(this.accent("multi-claude"))),
				this.dim(`├${horizontalLine}┤`),
				boxLine(`${this.busy}...`),
				this.dim(`╰${horizontalLine}╯`),
			);
			return lines;
		}

		lines.push(
			this.dim(`╭${horizontalLine}╮`),
			boxLine(this.bold(this.accent("multi-claude"))),
			this.dim(`├${horizontalLine}┤`),
		);

		if (this.loading) {
			lines.push(boxLine("loading..."));
		} else if (!this.rows.length) {
			lines.push(boxLine("no accounts"), boxLine(""), boxLine(this.dim("a  add account")));
		} else {
			for (let index = 0; index < this.rows.length; index++) {
				const row = this.rows[index];
				const isSelected = index === this.selectedIndex;

				const planLabel = row.plan ? theme.fg("accent", ` ${row.plan}`) : "";
				const cursor = isSelected ? theme.fg("accent", "▸ ") : "  ";
				const activeDot = row.active ? theme.fg("success", " ●") : "";
				lines.push(boxLine(`${cursor}${this.bold(row.email)}${planLabel}${activeDot}`));

				if (row.error) {
					lines.push(boxLine(this.dim(`   ${row.error}`)));
					continue;
				}

				for (const usageWindow of row.usageWindows) {
					const filled = Math.min(10, Math.round(usageWindow.percent / 10));
					const empty = 10 - filled;
					const bar = theme.fg(usageWindow.color, "█".repeat(filled)) + this.dim("░".repeat(empty));
					const resetLabel = usageWindow.reset
						? this.dim(` ${formatCountdown(new Date(usageWindow.reset))}`)
						: "";

					lines.push(
						boxLine(
							`   ${usageWindow.name.padEnd(6)} ${bar} ${usageWindow.percent.toFixed(0).padStart(3)}%${resetLabel}`,
						),
					);
				}
			}
		}

		lines.push(
			this.dim(`├${horizontalLine}┤`),
			boxLine(this.dim("↑↓ select  a add  ↵ switch  ⌫ remove  esc close")),
			this.dim(`╰${horizontalLine}╯`),
		);

		return lines;
	}
}

function getWeeklyReset(row: AccountRow): number {
	return row.usageWindows.find((window) => window.name === "week")?.reset ?? Number.POSITIVE_INFINITY;
}

function compareRowsByWeeklyReset(first: AccountRow, second: AccountRow): number {
	const resetDiff = getWeeklyReset(first) - getWeeklyReset(second);
	if (resetDiff !== 0) return resetDiff;
	return first.email.localeCompare(second.email);
}

function formatCountdown(date: Date): string {
	const diffMs = date.getTime() - Date.now();
	if (diffMs < 0) return "now";

	const minutes = Math.floor(diffMs / 60_000);
	if (minutes < 60) return `in ${minutes}m`;

	const hours = Math.floor(minutes / 60);
	const remainingMinutes = minutes % 60;
	const minuteSuffix = remainingMinutes ? ` ${remainingMinutes}m` : "";

	if (hours < 24) return `in ${hours}h${minuteSuffix}`;

	return `in ${Math.floor(hours / 24)}d`;
}

export default function (pi: ExtensionAPI) {
	pi.on("session_start", async (_event, context) => {
		const authStorage = context.modelRegistry.authStorage;
		const activeAccountKey = getActiveAccountKey(authStorage);

		for (const key of getAccountKeys(authStorage)) {
			try {
				await getAccessToken(authStorage, key);
				if (key === activeAccountKey) {
					const credential = authStorage.get(key);
					if (credential) authStorage.set(ACTIVE_KEY, credential);
				}
			} catch (error) {
				debug("session_start refresh failed", key, error);
			}
		}
	});

	pi.registerCommand("multi-claude", {
		description: "Manage multiple Anthropic Claude accounts",
		handler: async (_args, context) => {
			await context.ui.custom((tui, theme, keybindings, done) =>
				new AccountList(tui, theme, keybindings, done, context),
			);
		},
	});
}
