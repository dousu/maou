#!/usr/bin/env node

/**
 * Gradio UI Screenshot Script
 *
 * Captures screenshots of Gradio web interfaces using Playwright.
 * Optimized for Gradio's SPA architecture with proper wait strategies.
 *
 * Usage:
 *   node screenshot.js --url http://localhost:7860 --output /tmp/screenshot.png
 *   node screenshot.js --url http://localhost:7860 --base64
 *   node screenshot.js --url http://localhost:7860 --selector "#search-results"
 */

const { chromium } = require('playwright');

// Parse command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    url: 'http://localhost:7860',
    output: null,
    base64: false,
    selector: null,
    fullPage: false,
    waitFor: '.gradio-container',
    timeout: 30000,
    viewport: { width: 1280, height: 720 }
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--url':
        options.url = args[++i];
        break;
      case '--output':
        options.output = args[++i];
        break;
      case '--base64':
        options.base64 = true;
        break;
      case '--selector':
        options.selector = args[++i];
        break;
      case '--full-page':
        options.fullPage = true;
        break;
      case '--wait-for':
        options.waitFor = args[++i];
        break;
      case '--timeout':
        options.timeout = parseInt(args[++i], 10);
        break;
      case '--width':
        options.viewport.width = parseInt(args[++i], 10);
        break;
      case '--height':
        options.viewport.height = parseInt(args[++i], 10);
        break;
      case '--help':
        printHelp();
        process.exit(0);
    }
  }

  // Default output path if not base64 and no output specified
  if (!options.base64 && !options.output) {
    options.output = '/tmp/gradio-screenshot.png';
  }

  return options;
}

function printHelp() {
  console.log(`
Gradio UI Screenshot Script

Usage:
  node screenshot.js [options]

Options:
  --url <url>         Target URL (default: http://localhost:7860)
  --output <path>     Output file path (default: /tmp/gradio-screenshot.png)
  --base64            Output base64 to stdout instead of file
  --selector <sel>    CSS selector for specific element capture
  --full-page         Capture full scrollable page
  --wait-for <sel>    Wait for selector before capture (default: .gradio-container)
  --timeout <ms>      Navigation timeout in milliseconds (default: 30000)
  --width <px>        Viewport width (default: 1280)
  --height <px>       Viewport height (default: 720)
  --help              Show this help message

Examples:
  # Basic screenshot
  node screenshot.js --url http://localhost:7860 --output /tmp/gradio.png

  # Base64 output for Claude Vision API
  node screenshot.js --url http://localhost:7860 --base64

  # Capture specific element
  node screenshot.js --url http://localhost:7860 --selector "#search-results"

  # Full page screenshot with extended timeout
  node screenshot.js --url http://localhost:7860 --full-page --timeout 60000

Gradio UI Selectors:
  .gradio-container    Main container (default wait target)
  #mode-badge          Data mode display (MOCK/REAL)
  #id-search-input     Record ID search input
  #prev-page           Previous page button
  #next-page           Next page button
  #prev-record         Previous record button
  #next-record         Next record button
  #record-indicator    Current record display
`);
}

async function captureScreenshot(options) {
  let browser = null;

  try {
    // Launch headless Chromium
    browser = await chromium.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const context = await browser.newContext({
      viewport: options.viewport
    });

    const page = await context.newPage();

    // Navigate to the URL
    // Use 'domcontentloaded' instead of 'networkidle' because Gradio
    // keeps SSE connections open for live updates, preventing networkidle
    console.error(`Navigating to ${options.url}...`);
    await page.goto(options.url, {
      waitUntil: 'domcontentloaded',
      timeout: options.timeout
    });

    // Wait for Gradio container to be visible
    if (options.waitFor) {
      console.error(`Waiting for ${options.waitFor}...`);
      await page.waitForSelector(options.waitFor, {
        state: 'visible',
        timeout: options.timeout
      });
    }

    // Additional wait for Gradio's dynamic content to settle
    await page.waitForTimeout(500);

    // Determine what to capture
    let screenshotOptions = {
      type: 'png'
    };

    if (options.selector) {
      // Capture specific element
      console.error(`Capturing element: ${options.selector}`);
      const element = await page.$(options.selector);
      if (!element) {
        throw new Error(`Element not found: ${options.selector}`);
      }
      screenshotOptions.element = element;
    } else {
      // Full page or viewport
      screenshotOptions.fullPage = options.fullPage;
    }

    // Capture screenshot
    let screenshot;
    if (options.selector) {
      screenshot = await (await page.$(options.selector)).screenshot(screenshotOptions);
    } else {
      screenshot = await page.screenshot(screenshotOptions);
    }

    // Output handling
    if (options.base64) {
      // Output base64 to stdout
      const base64 = screenshot.toString('base64');
      process.stdout.write(base64);
      console.error('\nBase64 output written to stdout');
    } else {
      // Write to file
      const fs = require('fs');
      fs.writeFileSync(options.output, screenshot);
      console.error(`Screenshot saved to ${options.output}`);
    }

    return true;

  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

// Main execution
const options = parseArgs();
captureScreenshot(options);
