import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import type { Component } from '@earendil-works/pi-tui';
import extension from '../index.ts';

type Handler = (args: string, context: unknown) => Promise<void>;

function setup(auth: Record<string, unknown>) {
  const dir = mkdtempSync(join(tmpdir(), 'multi-claude-'));
  process.env['PI_CODING_AGENT_DIR'] = dir;
  const authPath = join(dir, 'auth.json');
  writeFileSync(authPath, JSON.stringify(auth));
  let handler: Handler | undefined;
  extension({
    on: () => {},
    registerCommand: (_name: string, options: { handler: Handler }) => {
      handler = options.handler;
    },
  } as never);
  return { authPath, handler: handler! };
}

function mockFetch() {
  globalThis.fetch = (async (url: string | URL) => {
    if (String(url).includes('/token')) {
      return new Response(
        JSON.stringify({
          access_token: 'new-access',
          refresh_token: 'new-refresh',
          expires_in: 3600,
        }),
      );
    }
    return new Response(JSON.stringify({}));
  }) as typeof fetch;
}

async function openList(handler: Handler, earlyInput?: string): Promise<Component> {
  let component: Component | undefined;
  const theme = { fg: (_c: string, t: string) => t, bold: (t: string) => t };
  const tui = { requestRender: () => {} };
  void handler('', {
    ui: {
      custom: (factory: (...args: unknown[]) => Component) => {
        component = factory(tui, theme, {}, () => {});
        if (earlyInput) component.handleInput?.(earlyInput);
        return new Promise(() => {});
      },
      notify: () => {},
    },
  });
  for (let i = 0; i < 100 && (!component || component.render(80).join().includes('loading')); i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  return component!;
}

test('keeps the active credential in sync when the list refreshes the active account', async () => {
  const expired = { type: 'oauth', access: 'old-access', refresh: 'old-refresh', expires: 0 };
  const { authPath, handler } = setup({ anthropic: expired, 'anthropic-0': expired });
  mockFetch();

  await openList(handler);

  const auth = JSON.parse(readFileSync(authPath, 'utf-8'));
  assert.equal(auth['anthropic-0'].access, 'new-access');
  assert.equal(auth.anthropic.access, 'new-access');
});

test('keeps a row selected when a key is pressed while loading', async () => {
  const valid = {
    type: 'oauth',
    access: 'access',
    refresh: 'refresh',
    expires: Date.now() + 3_600_000,
  };
  const { handler } = setup({ 'anthropic-0': valid });
  mockFetch();

  const component = await openList(handler, 'j');

  assert.match(component.render(80).join('\n'), /▸/);
});
