---
name: gradio-screenshot-capture
description: Capture screenshots of Gradio web UI using Playwright headless browser for visual feedback and verification. Use when testing UI rendering, debugging visual issues, or verifying component appearance.
---

# Gradio Screenshot Capture

Captures screenshots of Gradio web interfaces using Python Playwright headless browser. Optimized for Gradio's SPA architecture with proper wait strategies.

## Quick Start

Capture a screenshot of the running Gradio server:

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --output /tmp/gradio-screenshot.png
```

## Prerequisites

Ensure Playwright is installed:

```bash
poetry install --extras visualize
poetry run playwright install --with-deps chromium
```

## Instructions

### 1. Start Gradio Server

First, ensure the Gradio server is running:

```bash
poetry run maou visualize --use-mock-data &
sleep 10
```

### 2. Capture Screenshot

#### Basic Screenshot

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --output /tmp/gradio-screenshot.png
```

#### Base64 Output (for Claude Vision API)

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --base64
```

#### Capture Specific Element

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --selector "#search-results" \
  --output /tmp/search-results.png
```

#### Viewport Only Screenshot

By default, full page screenshots are captured. To capture only the visible viewport:

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --no-full-page \
  --output /tmp/viewport-only.png
```

### 3. View Screenshot

Use Claude Code's Read tool to view the captured screenshot:

```
Read the file /tmp/gradio-screenshot.png
```

Claude will analyze the screenshot and provide visual feedback.

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | http://localhost:7860 | Target URL |
| `--output`, `-o` | /tmp/gradio-screenshot.png | Output file path |
| `--base64` | false | Output base64 to stdout |
| `--selector`, `-s` | null | CSS selector for element capture |
| `--full-page` | true | Capture full scrollable page |
| `--no-full-page` | - | Capture viewport only |
| `--wait-for` | .gradio-container | Wait selector before capture |
| `--timeout` | 30000 | Navigation timeout (ms) |
| `--width` | 1280 | Viewport width |
| `--height` | 720 | Viewport height |
| `--settle-time` | 3000 | Wait time for dynamic content to stabilize (ms) |

## Gradio UI Selectors

Common selectors for the Maou visualization UI:

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

### UI Feedback Loop

1. Make changes to Gradio UI code
2. Restart Gradio server
3. Capture screenshot
4. Read screenshot with Claude
5. Get visual feedback and suggestions

### Automated Visual Testing

```bash
# Start server
poetry run maou visualize --use-mock-data &
sleep 10

# Capture multiple screenshots
poetry run maou screenshot \
  --url http://localhost:7860 \
  --output /tmp/main-view.png

poetry run maou screenshot \
  --url http://localhost:7860 \
  --selector "#mode-badge" \
  --output /tmp/mode-badge.png

# Stop server
lsof -ti :7860 | xargs kill -9 2>/dev/null || true
```

## Troubleshooting

### Playwright Not Installed

```bash
poetry install --extras visualize
poetry run playwright install --with-deps chromium
```

### Timeout Errors

Increase timeout for slow-loading pages:

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --timeout 60000
```

### Element Not Found

Verify the selector exists in the page:

```bash
# Try the default container first
poetry run maou screenshot \
  --url http://localhost:7860 \
  --wait-for "body"
```

### Server Not Running

Ensure Gradio server is running:

```bash
lsof -i :7860 || poetry run maou visualize --use-mock-data &
```

### Loading Screen Captured

If the screenshot shows "Loading..." instead of actual UI content, increase settle time:

```bash
poetry run maou screenshot \
  --url http://localhost:7860 \
  --settle-time 5000
```

## When to Use

- Testing Gradio UI changes
- Debugging visual rendering issues
- Verifying component appearance
- Creating visual documentation
- UI regression testing

## References

- [Playwright Python Documentation](https://playwright.dev/python/)
- [Gradio Documentation](https://www.gradio.app/)
- [Browser Automation Guide](../../../docs/browser-automation.md)
