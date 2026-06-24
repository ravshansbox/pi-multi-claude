# pi-multi-claude

Multi-account Anthropic Claude routing extension for pi.

## Install

```json
{
  "extensions": ["github:ravshansbox/pi-multi-claude"]
}
```

## Usage

Pi loads the extension from `./index.ts`. Run `/multi-claude` in the TUI to view saved Claude accounts, add a new account, switch the active account, or remove one.

For example, after adding two Claude accounts, open `/multi-claude`, move to the second account with `↑` or `↓`, and press `Enter` to switch pi to that account before continuing the session.

## Development

```bash
npm install
npm run typecheck
```
