const { app } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const HOST = process.env.OCR_HOST || '127.0.0.1';
const PORT = parseInt(process.env.OCR_PORT, 10) || 4891;

let serverProcess = null;
let startingPromise = null;

function getAppRoot() {
  return app.isPackaged ? process.resourcesPath : app.getAppPath();
}

function getPythonDir() {
  return path.join(getAppRoot(), 'python');
}

function getModelsDir() {
  return path.join(getAppRoot(), 'models');
}

function getServerUrl() {
  return `http://${HOST}:${PORT}`;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function checkHealth() {
  try {
    const res = await fetch(`${getServerUrl()}/health`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}

async function waitForHealth(timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await checkHealth()) return;
    await delay(300);
  }
  throw new Error('OCR server did not become healthy in time');
}

function getPythonCandidates() {
  const candidates = [];
  if (process.env.OCR_PYTHON) candidates.push(process.env.OCR_PYTHON);
  if (process.env.PYTHON_PATH) candidates.push(process.env.PYTHON_PATH);
  candidates.push('python');
  if (process.platform === 'win32') candidates.push('py');
  return candidates;
}

async function startServer() {
  if (await checkHealth()) return getServerUrl();
  if (serverProcess) return getServerUrl();
  if (startingPromise) return startingPromise;

  startingPromise = (async () => {
    const pythonDir = getPythonDir();
    const env = {
      ...process.env,
      OCR_MODELS_DIR: getModelsDir()
    };

    const args = [
      '-m', 'uvicorn', 'app:app',
      '--app-dir', pythonDir,
      '--host', HOST,
      '--port', String(PORT)
    ];

    let lastError = null;

    for (const cmd of getPythonCandidates()) {
      try {
        serverProcess = spawn(cmd, args, {
          env,
          cwd: getAppRoot(),
          windowsHide: true,
          stdio: ['ignore', 'pipe', 'pipe']
        });

        serverProcess.stdout?.on('data', (data) => {
          console.log(`[ocr] ${data.toString().trim()}`);
        });
        serverProcess.stderr?.on('data', (data) => {
          console.error(`[ocr] ${data.toString().trim()}`);
        });

        serverProcess.on('exit', (code, signal) => {
          console.warn(`[ocr] server exited code=${code} signal=${signal}`);
          serverProcess = null;
        });

        await waitForHealth();
        return getServerUrl();
      } catch (err) {
        lastError = err;
        if (serverProcess) {
          serverProcess.kill();
          serverProcess = null;
        }
      }
    }

    throw lastError || new Error('Failed to start OCR server');
  })();

  try {
    return await startingPromise;
  } finally {
    startingPromise = null;
  }
}

async function scanWithOcr({ imageDataUrl }) {
  if (!imageDataUrl) {
    throw new Error('Missing image data');
  }

  await startServer();

  const res = await fetch(`${getServerUrl()}/scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: imageDataUrl })
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`OCR server error (${res.status}): ${text}`);
  }

  return res.json();
}

function stopServer() {
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
}

module.exports = {
  startOcrServer: startServer,
  stopOcrServer: stopServer,
  scanWithOcr
};