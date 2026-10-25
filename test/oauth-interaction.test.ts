import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInteraction } from '../src/oauth-interaction.ts';

test('dismisses the input dialog when the prompt signal aborts', async () => {
  let dialogSignal: AbortSignal | undefined;
  const interaction = createInteraction(
    {
      notify: () => {},
      input: (_message, signal) => {
        dialogSignal = signal;
        return new Promise(() => {});
      },
    },
    new AbortController().signal,
  );
  const controller = new AbortController();
  void interaction
    .prompt({ type: 'manual_code', message: 'Paste code', signal: controller.signal })
    .catch(() => {});
  controller.abort();
  assert.equal(dialogSignal?.aborted, true);
});
