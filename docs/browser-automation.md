# Browser Automation Guide

This guide covers browser automation capabilities for the Maou project，specifically for capturing screenshots of the Gradio visualization UI.

## Overview

The project uses Playwright with Chromium for headless browser automation. This enables Claude Code to:

- Capture screenshots of the Gradio UI for visual feedback
- Verify UI rendering after code changes
- Debug visual issues with AI-assisted analysis
- Create visual documentation

## Installation

### Automatic Installation (DevContainer)

When using GitHub Codespaces or DevContainer，Playwright is automatically installed via `postCreateCommand`:

```bash
cd scripts/browser && npm install && npx playwright install --with-deps chromium
```

### Manual Installation

If you need to install manually:

```bash
# Navigate to browser scripts directory
cd /workspaces/maou/scripts/browser

# Install npm dependencies
npm install

# Install Chromium browser
npx playwright install --with-deps chromium
```

### Verify Installation

```bash
npx playwright --version
```

Expected output: `Version 1.40.x`

## Screenshot Script

The screenshot script is located at `scripts/browser/screenshot.js`.

### Basic Usage

```bash
# Basic screenshot
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --output /tmp/screenshot.png

# Base64 output for Claude Vision API
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --base64

# Capture specific element
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --selector "#search-results"
```

### Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | http://localhost:7860 | Target URL |
| `--output` | /tmp/gradio-screenshot.png | Output file path |
| `--base64` | false | Output base64 to stdout |
| `--selector` | null | CSS selector for element capture |
| `--full-page` | false | Capture full scrollable page |
| `--wait-for` | .gradio-container | Wait selector before capture |
| `--timeout` | 30000 | Navigation timeout (ms) |
| `--width` | 1280 | Viewport width |
| `--height` | 720 | Viewport height |

## Gradio UI Selectors

The Maou visualization UI exposes these common selectors:

| Selector | Description |
|----------|-------------|
| `.gradio-container` | Main container (default wait target) |
| `#mode-badge` | Data mode display (MOCK/REAL) |
| `#id-search-input` | Record ID search input |
| `#prev-page` | Previous page button |
| `#next-page` | Next page button |
| `#prev-record` | Previous record button |
| `#next-record` | Next record button |
| `#record-indicator` | Current record display |

## Workflow Examples

### Complete UI Feedback Loop

```bash
# 1. Start Gradio server
poetry run maou visualize --use-mock-data &
sleep 10

# 2. Capture screenshot
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --output /tmp/gradio-screenshot.png

# 3. View with Claude (Read tool reads PNG files)
# Claude will analyze the screenshot visually

# 4. Stop server
lsof -ti :7860 | xargs kill -9 2>/dev/null || true
```

### Capturing Multiple Views

```bash
# Main view
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --output /tmp/main-view.png

# Mode badge only
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --selector "#mode-badge" \
  --output /tmp/mode-badge.png

# Full page with scroll
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --full-page \
  --output /tmp/full-page.png
```

### Using with Claude Code Skill

The `gradio-screenshot-capture` skill automates this workflow. Trigger with:

- "capture screenshot"
- "screenshot UI"
- "visual feedback"
- "スクリーンショット"

## Gradio Wait Strategies

Gradio is a Single Page Application (SPA) that loads content dynamically and keeps SSE connections open for live updates. The script uses:

1. **domcontentloaded**: Wait for DOM to be ready (not networkidle, which never triggers due to SSE)
2. **Selector wait**: Wait for `.gradio-container` to be visible
3. **Buffer time**: Additional 500ms for dynamic content

For slow-loading pages，increase timeout:

```bash
node /workspaces/maou/scripts/browser/screenshot.js \
  --url http://localhost:7860 \
  --timeout 60000
```

## Troubleshooting

### Playwright Not Found

```bash
cd /workspaces/maou/scripts/browser
npm install
npx playwright install --with-deps chromium
```

### Browser Launch Failure

Ensure system dependencies are installed:

```bash
sudo apt-get update
sudo apt-get install -y libnss3 libnspr4 libasound2
```

Or reinstall with dependencies:

```bash
npx playwright install --with-deps chromium
```

### Timeout Errors

1. Verify server is running: `lsof -i :7860`
2. Increase timeout: `--timeout 60000`
3. Use simpler wait target: `--wait-for "body"`

### Element Not Found

1. Verify selector in browser DevTools
2. Check if element is dynamically loaded
3. Add explicit wait time after navigation

### Memory Issues in DevContainer

Chromium can be memory-intensive. If encountering OOM:

```bash
# Run with reduced memory
npx playwright install chromium  # Skip --with-deps if low memory
```

## Architecture

```
scripts/browser/
├── package.json       # Playwright dependency
└── screenshot.js      # Screenshot capture script
```

**Design decisions:**

1. **Chromium only**: Minimal footprint (~150MB)
2. **Node.js isolated**: npm dependencies in `scripts/browser/`，no impact on Poetry
3. **Gradio-optimized**: Default wait for `.gradio-container`
4. **Claude Vision ready**: Base64 output option

## References

- [Playwright Documentation](https://playwright.dev/)
- [Gradio Documentation](https://www.gradio.app/)
- [Skill Documentation](../.claude/skills/gradio-screenshot-capture/SKILL.md)
