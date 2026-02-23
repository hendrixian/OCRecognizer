// main.js - Electron main process
const { app, BrowserWindow, Menu, shell, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { startOcrServer, stopOcrServer, scanWithOcr } = require('./electron/ocrServer');

// Keep a global reference of the window object
let mainWindow;

// Check if we're in development mode
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
const iconPath = isDev
  ? path.join(__dirname, 'public', 'favicon.ico')
  : path.join(__dirname, 'build', 'favicon.ico');

function createWindow() {
  // Create the browser window with native Windows styling
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    icon: iconPath, // Use existing icon (build in prod, public in dev)
    webPreferences: {
      nodeIntegration: false, // Security: don't allow node in renderer
      contextIsolation: true, // Security: isolate contexts
      preload: path.join(__dirname, 'preload.js') // Secure bridge
    },
    show: false, // Don't show until ready
    frame: true, // Native Windows frame
    titleBarStyle: 'default',
    backgroundColor: '#062015'
  });

  // Load the app
  if (isDev) {
    // In development, load from React dev server
    mainWindow.loadURL('http://localhost:3000');
    // Open DevTools
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built files
    mainWindow.loadFile(path.join(__dirname, 'build', 'index.html'));
  }

  // Show window when ready to prevent white flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Handle external links (open in default browser)
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https:') || url.startsWith('http:')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  // Emitted when the window is closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Create native application menu
  const menuTemplate = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Photo...',
          accelerator: 'Ctrl+O',
          click: () => {
            mainWindow.webContents.send('menu-open-photo');
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: 'Alt+F4',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About NRC Scanner',
          click: () => {
            // You can create an about dialog here
            shell.openExternal('https://github.com/yourusername/nrc-scanner');
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(menuTemplate);
  Menu.setApplicationMenu(menu);
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  createWindow();

  startOcrServer().catch((err) => {
    console.error('[ocr] Failed to start OCR server:', err);
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopOcrServer();
});

// Handle camera/microphone permissions (important for your scanner)
app.on('session-created', (session) => {
  session.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media' || permission === 'camera' || permission === 'microphone') {
      // Grant camera/microphone access
      callback(true);
    } else {
      callback(false);
    }
  });
});

// App info
ipcMain.handle('get-app-version', () => app.getVersion());

// OCR scan
ipcMain.handle('scan-nrc', async (event, payload) => {
  try {
    const data = await scanWithOcr(payload || {});
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: err?.message || 'Scan failed' };
  }
});

// Handle file system operations (if needed)
ipcMain.handle('select-file', async () => {
  const { dialog } = require('electron');
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif'] }
    ]
  });
  
  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

// Save scanned data to file (optional feature)
ipcMain.handle('save-scanned-data', async (event, data) => {
  const { dialog } = require('electron');
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Scanned Data',
    defaultPath: `nrc-data-${Date.now()}.json`,
    filters: [
      { name: 'JSON Files', extensions: ['json'] },
      { name: 'Text Files', extensions: ['txt'] }
    ]
  });
  
  if (!result.canceled) {
    fs.writeFileSync(result.filePath, JSON.stringify(data, null, 2));
    return true;
  }
  return false;
});