// preload.js - Secure bridge between React and Electron
const { contextBridge, ipcRenderer } = require('electron');

const isDev = process.env.NODE_ENV === 'development';

// Expose protected methods that allow the renderer process to use
// specific electron APIs without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // File selection
  selectFile: () => ipcRenderer.invoke('select-file'),
  
  // Save data
  saveScannedData: (data) => ipcRenderer.invoke('save-scanned-data', data),
  
  // Menu events
  onMenuOpenPhoto: (callback) => {
    ipcRenderer.on('menu-open-photo', () => callback());
  },
  
  // App info
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),

  // OCR scan
  scanNrc: (payload) => ipcRenderer.invoke('scan-nrc', payload),
  
  // Platform info
  isWindows: process.platform === 'win32'
});

// Expose any environment info needed
contextBridge.exposeInMainWorld('appInfo', {
  platform: process.platform,
  isDev
});