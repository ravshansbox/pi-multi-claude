import type { Credential } from '@earendil-works/pi-ai';
import { getAgentDir } from '@earendil-works/pi-coding-agent';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import lockfile from 'proper-lockfile';

const WRITE_OPTIONS = { encoding: 'utf-8' as const, mode: 0o600 };
const LOCK_ATTEMPTS = 10;
const LOCK_DELAY_MS = 20;

type AuthData = Record<string, Credential>;

export class AccountStore {
  private readonly authPath: string;

  constructor(authPath?: string) {
    this.authPath = authPath ?? join(getAgentDir(), 'auth.json');
  }

  get(key: string): Credential | undefined {
    return this.readAll()[key];
  }

  list(): string[] {
    return Object.keys(this.readAll());
  }

  set(key: string, credential: Credential): void {
    this.modify((data) => {
      data[key] = credential;
    });
  }

  remove(key: string): void {
    this.modify((data) => {
      delete data[key];
    });
  }

  private readAll(): AuthData {
    if (!existsSync(this.authPath)) return {};
    const raw = readFileSync(this.authPath, 'utf-8').trim();
    if (!raw) return {};
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      throw new Error(
        `Failed to read auth.json: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new Error('Invalid auth.json: expected an object');
    }
    return parsed as AuthData;
  }

  private modify(fn: (data: AuthData) => void): void {
    this.ensureFile();
    const release = this.acquireLock();
    try {
      const data = this.readAll();
      fn(data);
      writeFileSync(
        this.authPath,
        `${JSON.stringify(data, null, 2)}\n`,
        WRITE_OPTIONS,
      );
      chmodSync(this.authPath, 0o600);
    } finally {
      release();
    }
  }

  private ensureFile(): void {
    const dir = dirname(this.authPath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true, mode: 0o700 });
    if (!existsSync(this.authPath)) {
      writeFileSync(this.authPath, '{}', WRITE_OPTIONS);
      chmodSync(this.authPath, 0o600);
    }
  }

  private acquireLock(): () => void {
    let lastError: unknown;
    for (let attempt = 1; attempt <= LOCK_ATTEMPTS; attempt++) {
      try {
        return lockfile.lockSync(this.authPath, { realpath: false });
      } catch (error) {
        const code =
          typeof error === 'object' && error !== null && 'code' in error
            ? String((error as { code: unknown }).code)
            : undefined;
        if (code !== 'ELOCKED' || attempt === LOCK_ATTEMPTS) throw error;
        lastError = error;
        const start = Date.now();
        while (Date.now() - start < LOCK_DELAY_MS) {
          /* spin so callers stay synchronous */
        }
      }
    }
    throw lastError ?? new Error('Failed to acquire auth.json lock');
  }
}
