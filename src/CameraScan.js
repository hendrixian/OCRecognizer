import React, { useEffect, useRef, useState } from 'react';
import './CameraScan.css';
import { scanNrcFromDataUrl } from './services/ocrService';
import { toUiScanResult, DEFAULT_SCAN_RESULT } from './services/scanResultAdapter';

const SCAN_INTERVAL_MS = 500;
const MAX_CAPTURE_WIDTH = 1280;

const CameraScan = ({ onBack, onScanComplete }) => {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const streamRef = useRef(null);
  const scanStateRef = useRef({ inFlight: false, lastScan: 0 });

  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDetecting, setIsDetecting] = useState(true);
  const [status, setStatus] = useState('Idle');
  const [error, setError] = useState('');
  const [lastResult, setLastResult] = useState(DEFAULT_SCAN_RESULT);
  const [videoAspect, setVideoAspect] = useState(16 / 9);
  const [lastUpdated, setLastUpdated] = useState(null);

  const nrcNumberDisplay =
    lastResult?.nrcNumber ||
    lastResult?.nrcNumberBurmese ||
    lastResult?.rawDigits ||
    '';

  const clearOverlay = () => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  };

  const ensureCanvasSize = (width, height) => {
    const capture = captureRef.current;
    const overlay = overlayRef.current;
    if (!capture || !overlay) return;
    if (capture.width !== width || capture.height !== height) {
      capture.width = width;
      capture.height = height;
    }
    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }
  };

  const drawBoxes = (boxes = []) => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!Array.isArray(boxes) || boxes.length === 0) return;

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#F8F3CE';
    ctx.fillStyle = 'rgba(248, 243, 206, 0.1)';
    ctx.font = '14px "Inter", system-ui, sans-serif';

    boxes.forEach((box) => {
      const x = Math.max(0, box.x1);
      const y = Math.max(0, box.y1);
      const w = Math.max(0, box.x2 - box.x1);
      const h = Math.max(0, box.y2 - box.y1);

      ctx.strokeRect(x, y, w, h);

      const cls = typeof box.cls === 'number' ? Math.round(box.cls) : null;
      const conf = typeof box.conf === 'number' ? Math.round(box.conf * 100) : null;
      const label = cls !== null ? `digit_${cls}${conf !== null ? ` ${conf}%` : ''}` : '';

      if (label) {
        const padding = 4;
        const textWidth = ctx.measureText(label).width;
        const textX = x;
        const textY = Math.max(0, y - 18);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillRect(textX, textY, textWidth + padding * 2, 18);
        ctx.fillStyle = '#F8F3CE';
        ctx.fillText(label, textX + padding, textY + 13);
      }
    });
  };

  const refreshDevices = async () => {
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      const cameras = list.filter((device) => device.kind === 'videoinput');
      setDevices(cameras);
      if (!selectedDeviceId && cameras.length > 0) {
        setSelectedDeviceId(cameras[0].deviceId);
      }
    } catch (err) {
      setError(err?.message || 'Unable to list cameras');
    }
  };

  const stopStream = () => {
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    streamRef.current = null;
    setIsStreaming(false);
    setStatus('Stopped');
    clearOverlay();
  };

  const startStream = async (deviceIdOverride) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera API not available in this environment.');
      return;
    }

    setError('');
    const deviceId = deviceIdOverride || selectedDeviceId;
    const constraints = {
      audio: false,
      video: deviceId
        ? { deviceId: { exact: deviceId } }
        : { facingMode: 'environment' }
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsStreaming(true);
      setStatus('Live');
      await refreshDevices();
    } catch (err) {
      setError(err?.message || 'Failed to access camera');
    }
  };

  const handleDeviceChange = async (event) => {
    const nextId = event.target.value;
    setSelectedDeviceId(nextId);
    if (isStreaming) {
      stopStream();
      await startStream(nextId);
    }
  };

  const captureFrame = () => {
    const video = videoRef.current;
    const capture = captureRef.current;
    const overlay = overlayRef.current;
    if (!video || !capture || !overlay) return null;

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    if (!videoWidth || !videoHeight) return null;

    const scale = Math.min(1, MAX_CAPTURE_WIDTH / videoWidth);
    const width = Math.round(videoWidth * scale);
    const height = Math.round(videoHeight * scale);
    ensureCanvasSize(width, height);

    const ctx = capture.getContext('2d');
    ctx.drawImage(video, 0, 0, width, height);
    return capture.toDataURL('image/jpeg', 0.85);
  };

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video?.videoHeight) return;
    setVideoAspect(video.videoWidth / video.videoHeight);
  };

  const handleUseResult = () => {
    if (!onScanComplete || !nrcNumberDisplay) return;
    onScanComplete(lastResult);
  };

  useEffect(() => {
    refreshDevices();
    return () => {
      stopStream();
    };
  }, []);

  useEffect(() => {
    if (!isStreaming) return;
    let cancelled = false;

    const loop = async () => {
      if (cancelled) return;

      if (isDetecting) {
        const now = Date.now();
        if (!scanStateRef.current.inFlight && now - scanStateRef.current.lastScan >= SCAN_INTERVAL_MS) {
          const dataUrl = captureFrame();
          if (dataUrl) {
            scanStateRef.current.inFlight = true;
            scanStateRef.current.lastScan = now;
            try {
              const apiResult = await scanNrcFromDataUrl(dataUrl);
              const uiResult = toUiScanResult(apiResult);
              setLastResult(uiResult);
              setLastUpdated(new Date());
              drawBoxes(apiResult?.boxes || []);
            } catch (err) {
              setError(err?.message || 'Scan failed');
              setIsDetecting(false);
            } finally {
              scanStateRef.current.inFlight = false;
            }
          }
        }
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);

    return () => {
      cancelled = true;
    };
  }, [isStreaming, isDetecting]);

  return (
    <div className="camera-scan-container">
      <div className="camera-card">
        <div className="camera-header">
          <button className="back-button" onClick={onBack}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="#F8F3CE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h1 className="camera-title">NRC SCANNER</h1>
          <div className="title-underline"></div>
        </div>

        <div className="camera-subtitle">
          Use your iPhone camera via Camo. Select the camera device below.
        </div>

        <div className="camera-controls">
          <div className="camera-select">
            <label htmlFor="camera-device">Camera</label>
            <select
              id="camera-device"
              value={selectedDeviceId}
              onChange={handleDeviceChange}
              disabled={devices.length === 0}
            >
              {devices.map((device, idx) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${idx + 1}`}
                </option>
              ))}
            </select>
            <button className="camera-link-btn" onClick={refreshDevices}>
              Refresh
            </button>
          </div>

          <div className="camera-actions">
            {!isStreaming ? (
              <button className="primary-btn" onClick={() => startStream()}>
                Start Camera
              </button>
            ) : (
              <button className="secondary-btn" onClick={stopStream}>
                Stop Camera
              </button>
            )}

            <button
              className="scan-toggle-btn"
              onClick={() => setIsDetecting((prev) => !prev)}
              disabled={!isStreaming}
            >
              {isDetecting ? 'Pause Scan' : 'Resume Scan'}
            </button>
          </div>
        </div>

        <div className="camera-stage" style={{ aspectRatio: videoAspect }}>
          <video
            ref={videoRef}
            className="camera-video"
            onLoadedMetadata={handleLoadedMetadata}
            autoPlay
            muted
            playsInline
          />
          <canvas ref={overlayRef} className="camera-overlay" />
          <canvas ref={captureRef} className="camera-capture" />
          {!isStreaming && (
            <div className="camera-placeholder">
              Start the camera to begin real-time scanning.
            </div>
          )}
        </div>

        <div className="camera-status">
          <div className={`status-pill ${isStreaming ? 'live' : ''}`}>
            {status}
          </div>
          <div className="status-text">
            {isStreaming ? (isDetecting ? 'Scanning…' : 'Scan paused') : 'Camera off'}
          </div>
        </div>

        <div className="camera-results">
          <div className="result-block">
            <div className="info-label">NRC Number</div>
            <div className="info-value">{nrcNumberDisplay || '—'}</div>
          </div>
          <div className="result-block">
            <div className="info-label">Confidence</div>
            <div className="info-value">
              {typeof lastResult.confidence === 'number'
                ? `${Math.round(lastResult.confidence * 100)}%`
                : '—'}
            </div>
          </div>
          <div className="result-block">
            <div className="info-label">Last Update</div>
            <div className="info-value">
              {lastUpdated ? lastUpdated.toLocaleTimeString() : '—'}
            </div>
          </div>
        </div>

        <div className="camera-footer">
          <button
            className="secondary-btn"
            onClick={handleUseResult}
            disabled={!nrcNumberDisplay}
          >
            Use Current Result
          </button>
          <button
            className="secondary-btn"
            onClick={() => navigator.clipboard.writeText(nrcNumberDisplay)}
            disabled={!nrcNumberDisplay}
          >
            Copy Number
          </button>
        </div>

        {error && <div className="camera-error">{error}</div>}
      </div>
    </div>
  );
};

export default CameraScan;
