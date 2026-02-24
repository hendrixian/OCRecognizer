import React, { useEffect, useRef } from 'react';
import './ScanResult.css';

const ScanResult = ({ onBack, onNewScan, scannedData, scannedImage }) => {
  const imageRef = useRef(null);
  const overlayRef = useRef(null);

  // Mock data - this will come from your backend/ML model
  const resultData = scannedData || {
    nrcNumber: "12/OUKAMA(N)123456",
    name: "THANT ZIN",
    birthDate: "15.05.1990",
    fatherName: "U MYA",
    motherName: "Daw AYE",
    address: "No.123, Yangon-Insein Road, Hlaing Township, Yangon",
    issueDate: "20.01.2020",
    expiryDate: "19.01.2030",
    confidence: 0.95
  };
  const nrcNumberDisplay =
    resultData.nrcNumber ||
    resultData.nrcNumberBurmese ||
    resultData.rawDigits ||
    '';
  const hasImage = Boolean(scannedImage);
  const regionBoxes = resultData?.regionBoxes || resultData?.areaBoxes || [];
  const digitBoxes = resultData?.boxes || [];

  const buildDigitLabel = (box) => {
    const cls = typeof box.cls === 'number' ? Math.round(box.cls) : null;
    const conf = typeof box.conf === 'number' ? Math.round(box.conf * 100) : null;
    if (cls === null) return '';
    return `${cls}${conf !== null ? ` ${conf}%` : ''}`;
  };

  const buildRegionLabel = (box) => {
    const label = typeof box.label === 'string' ? box.label : '';
    const cls = typeof box.cls === 'number' ? Math.round(box.cls) : null;
    const conf = typeof box.conf === 'number' ? Math.round(box.conf * 100) : null;
    const base = label || (cls !== null ? `region_${cls}` : '');
    if (!base) return '';
    return `${base}${conf !== null ? ` ${conf}%` : ''}`;
  };

  const drawBoxes = (ctx, boxes = [], scaleX, scaleY, options = {}) => {
    if (!Array.isArray(boxes) || boxes.length === 0) return;

    const {
      strokeStyle = '#F8F3CE',
      lineWidth = 2,
      font = '14px "Inter", system-ui, sans-serif',
      labelColor = '#F8F3CE',
      labelBackground = 'rgba(0, 0, 0, 0.55)',
      labelBuilder
    } = options;

    ctx.save();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;
    ctx.font = font;

    boxes.forEach((box) => {
      const x1 = Math.max(0, box.x1 * scaleX);
      const y1 = Math.max(0, box.y1 * scaleY);
      const x2 = Math.max(0, box.x2 * scaleX);
      const y2 = Math.max(0, box.y2 * scaleY);
      const w = Math.max(0, x2 - x1);
      const h = Math.max(0, y2 - y1);

      ctx.strokeRect(x1, y1, w, h);

      const label = typeof labelBuilder === 'function' ? labelBuilder(box) : '';
      if (label) {
        const padding = 4;
        const textWidth = ctx.measureText(label).width;
        const textX = x1;
        const textY = Math.max(0, y1 - 18);
        ctx.fillStyle = labelBackground;
        ctx.fillRect(textX, textY, textWidth + padding * 2, 18);
        ctx.fillStyle = labelColor;
        ctx.fillText(label, textX + padding, textY + 13);
      }
    });

    ctx.restore();
  };

  const drawOverlay = () => {
    const img = imageRef.current;
    const canvas = overlayRef.current;
    if (!img || !canvas) return;

    const { clientWidth, clientHeight, naturalWidth, naturalHeight } = img;
    if (!clientWidth || !clientHeight || !naturalWidth || !naturalHeight) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(clientWidth * dpr);
    canvas.height = Math.round(clientHeight * dpr);
    canvas.style.width = `${clientWidth}px`;
    canvas.style.height = `${clientHeight}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, clientWidth, clientHeight);

    if ((!regionBoxes || regionBoxes.length === 0) && (!digitBoxes || digitBoxes.length === 0)) {
      return;
    }

    const scaleX = clientWidth / naturalWidth;
    const scaleY = clientHeight / naturalHeight;

    drawBoxes(ctx, regionBoxes, scaleX, scaleY, {
      strokeStyle: '#6EF2C4',
      lineWidth: 3,
      labelColor: '#061A12',
      labelBackground: 'rgba(110, 242, 196, 0.75)',
      labelBuilder: buildRegionLabel
    });

    drawBoxes(ctx, digitBoxes, scaleX, scaleY, {
      strokeStyle: '#F8F3CE',
      lineWidth: 1.5,
      labelBuilder: buildDigitLabel
    });
  };

  useEffect(() => {
    if (!hasImage) return;
    const handleResize = () => drawOverlay();
    window.addEventListener('resize', handleResize);
    drawOverlay();
    return () => window.removeEventListener('resize', handleResize);
  }, [hasImage, scannedImage, scannedData]);

  return (
    <div className="result-container">
      <div className="result-card">
        {/* Header with back button and title */}
        <div className="result-header">
          <button className="back-button" onClick={onBack}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="#F8F3CE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h1 className="result-title">NRC SCANNER</h1>
          <div className="title-underline"></div>
        </div>

        {/* Confidence Badge */}
        <div className="confidence-badge">
          <span className="confidence-text">
            Confidence: {(resultData.confidence * 100).toFixed(0)}%
          </span>
        </div>

        {/* Main content - SHOW SCANNED TEXT HERE */}
        <div className="scanned-text-header">
          <h2>SHOW SCANNED TEXT HERE</h2>
        </div>

        <div className={`result-body ${hasImage ? 'with-image' : 'no-image'}`}>
          {hasImage && (
            <div className="result-image-panel">
              <div className="result-image-label">Scanned Image</div>
              <div className="result-image-frame">
                <div className="result-image-wrapper">
                  <img
                    ref={imageRef}
                    src={scannedImage}
                    alt="Scanned document"
                    className="result-image"
                    onLoad={drawOverlay}
                  />
                  <canvas ref={overlayRef} className="result-image-overlay" />
                </div>
              </div>
            </div>
          )}

          <div className="result-info-panel">
            {/* Scanned Information Display */}
            <div className="scanned-info">
              {/* NRC Number - Most important */}
              <div className="info-card highlight">
                <div className="info-label">NRC Number</div>
                <div className="info-value">{nrcNumberDisplay}</div>
                <div className="info-copy" onClick={() => navigator.clipboard.writeText(nrcNumberDisplay)}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="4" y="8" width="12" height="12" rx="1" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M8 4V6H16V16H18V4H8Z" fill="currentColor"/>
                  </svg>
                </div>
              </div>

              {/* Personal Information Grid */}
              <div className="info-grid">
                <div className="info-item">
                  <div className="info-label">Full Name</div>
                  <div className="info-value">{resultData.name}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Date of Birth</div>
                  <div className="info-value">{resultData.birthDate}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Father's Name</div>
                  <div className="info-value">{resultData.fatherName}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Mother's Name</div>
                  <div className="info-value">{resultData.motherName}</div>
                </div>
              </div>

              {/* Address - Full width */}
              <div className="info-item full-width">
                <div className="info-label">Address</div>
                <div className="info-value">{resultData.address}</div>
              </div>

              {/* Date Information Grid */}
              <div className="info-grid">
                <div className="info-item">
                  <div className="info-label">Issue Date</div>
                  <div className="info-value">{resultData.issueDate}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Expiry Date</div>
                  <div className="info-value">{resultData.expiryDate}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="result-actions">
          <button className="primary-btn" onClick={() => {/* Handle save/export */}}>
            Save Information
          </button>
          <button className="secondary-btn" onClick={onNewScan}>
            Scan New Document
          </button>
        </div>

        {/* Raw JSON Data (for debugging) - Remove in production */}
        <details className="raw-data">
          <summary>Raw JSON Data</summary>
          <pre>{JSON.stringify(resultData, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
};

export default ScanResult;
