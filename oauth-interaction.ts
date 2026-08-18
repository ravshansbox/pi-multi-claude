import type {
  AuthEvent,
  AuthPrompt,
  ProviderAuthInteraction,
} from '@earendil-works/pi-ai';

export interface InteractionHost {
  notify(message: string, level: 'info' | 'error'): void;
  input(message: string): Promise<string | undefined>;
}

function openInBrowser(url: string): void {
  void import('node:child_process').then(({ exec }) => {
    let openCmd: string;
    if (process.platform === 'darwin') {
      openCmd = `open '${url}'`;
    } else if (process.platform === 'win32') {
      openCmd = `start "" "${url}"`;
    } else {
      openCmd = `xdg-open '${url}'`;
    }
    exec(openCmd);
  });
}

export function createInteraction(
  host: InteractionHost,
  signal: AbortSignal,
): ProviderAuthInteraction {
  return {
    signal,
    notify(event: AuthEvent): void {
      switch (event.type) {
        case 'auth_url':
          host.notify(`Open: ${event.url}`, 'info');
          if (event.instructions) host.notify(event.instructions, 'info');
          openInBrowser(event.url);
          return;
        case 'device_code':
          host.notify(
            `Enter code ${event.userCode} at ${event.verificationUri}`,
            'info',
          );
          openInBrowser(event.verificationUri);
          return;
        case 'info':
        case 'progress':
          host.notify(event.message, 'info');
          return;
        default:
          return;
      }
    },
    async prompt(prompt: AuthPrompt): Promise<string> {
      if (prompt.type === 'select') {
        const first = prompt.options[0];
        if (!first) throw new Error('No options offered');
        return first.id;
      }
      const answer = await host.input(prompt.message);
      if (!answer?.trim()) throw new Error('Cancelled');
      return answer.trim();
    },
  };
}
