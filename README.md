# pi-multi-claude

Multi-account Anthropic Claude routing extension for pi.

## Install

```bash
pi install npm:@ravshansbox/pi-multi-claude
```

Add `-l` to install it in project settings.

## Usage

Pi loads the extension from `./index.ts`. Run `/multi-claude` in the TUI to view saved Claude accounts, add a new account, switch the active account, or remove one.

For example, after adding two Claude accounts, open `/multi-claude`, move to the second account with `↑` or `↓`, and press `Enter` to switch pi to that account before continuing the session.

## Configuration

Set `MULTI_CLAUDE_DEBUG=1` to print debug logs to stderr.

## Development

```bash
npm install
npm run check
```
