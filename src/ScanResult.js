import React, { useEffect, useRef, useState } from 'react';
import './ScanResult.css';
import { getBurmeseClassLabel } from './utils/burmeseClassLabels';
import { deriveNrcNumberFromDetections } from './utils/nrcNumber';

const ScanResult = ({ onBack, onNewScan, scannedData, scannedImage }) => {
  const imageRef = useRef(null);
  const overlayRef = useRef(null);
  const overlayStateRef = useRef({ region: [], digit: [] });

  const [isZoomed, setIsZoomed] = useState(false);
  const [hoveredBox, setHoveredBox] = useState(null);

  // Mock data - this will come from your backend/ML model
  const resultData = scannedData || {
    nrcNumber: "12/OUKAMA(N)123456",
    name: "THANT ZIN",
    birthDate: "15.05.1990",
    fatherName: "U MYA",
    motherName: "Daw AYE",
    religion: "",
    height: "",
    distinctFeature: "",
    address: "No.123, Yangon-Insein Road, Hlaing Township, Yangon",
    issueDate: "20.01.2020",
    expiryDate: "19.01.2030",
    confidence: 0.95
  };
  const hasImage = Boolean(scannedImage);
  const regionBoxes = resultData?.regionBoxes || resultData?.areaBoxes || [];
  const digitBoxes = resultData?.boxes || [];
  const toText = (value) => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' && Number.isFinite(value)) return String(value);
    return '';
  };
  const derivedNrcNumber = deriveNrcNumberFromDetections(digitBoxes, regionBoxes);
  const nrcNumberDisplay =
    toText(resultData.nrcNumber) ||
    toText(resultData.nrc_number) ||
    toText(resultData.nrcNumberLatin) ||
    toText(resultData.nrc_number_latin) ||
    toText(resultData.nrcNumberBurmese) ||
    toText(resultData.nrc_number_burmese) ||
    toText(resultData.rawDigits) ||
    toText(resultData.raw_digits) ||
    derivedNrcNumber ||
    '';

  const buildDigitLabel = (box) => {
    const cls = typeof box.cls === 'number' ? box.cls : null;
    const conf = typeof box.conf === 'number' ? Math.round(box.conf * 100) : null;
    if (cls === null) return '';
    const label = getBurmeseClassLabel(cls);
    if (!label) return '';
    return `${label}${conf !== null ? ` ${conf}%` : ''}`;
  };

  const buildRegionLabel = (box) => {
    const label = typeof box.label === 'string' ? box.label : '';
    const cls = typeof box.cls === 'number' ? Math.round(box.cls) : null;
    const conf = typeof box.conf === 'number' ? Math.round(box.conf * 100) : null;
    const base = label || (cls !== null ? `region_${cls}` : '');
    if (!base) return '';
    return `${base}${conf !== null ? ` ${conf}%` : ''}`;
  };

  const buildScaledBoxes = (boxes, scaleX, scaleY) =>
    boxes.map((box, index) => {
      const x1 = Math.max(0, box.x1 * scaleX);
      const y1 = Math.max(0, box.y1 * scaleY);
      const x2 = Math.max(0, box.x2 * scaleX);
      const y2 = Math.max(0, box.y2 * scaleY);
      return {
        index,
        box,
        x1,
        y1,
        x2,
        y2
      };
    });

  const drawScaledBoxes = (ctx, scaledBoxes = [], options = {}) => {
    if (!Array.isArray(scaledBoxes) || scaledBoxes.length === 0) return;

    const {
      strokeStyle = '#F8F3CE',
      lineWidth = 2
    } = options;

    ctx.save();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;

    scaledBoxes.forEach((item) => {
      const w = Math.max(0, item.x2 - item.x1);
      const h = Math.max(0, item.y2 - item.y1);
      if (w === 0 || h === 0) return;
      ctx.strokeRect(item.x1, item.y1, w, h);
    });

    ctx.restore();
  };

  const drawHoverLabel = (ctx, item, options = {}) => {
    if (!item) return;

    const {
      strokeStyle = '#F8F3CE',
      lineWidth = 2,
      font = '14px "Myanmar Text", "Noto Sans Myanmar", "Pyidaungsu", "Inter", system-ui, sans-serif',
      labelColor = '#F8F3CE',
      labelBackground = 'rgba(0, 0, 0, 0.55)',
      labelBuilder
    } = options;

    const label = typeof labelBuilder === 'function' ? labelBuilder(item.box) : '';
    if (!label) return;

    const w = Math.max(0, item.x2 - item.x1);
    const h = Math.max(0, item.y2 - item.y1);

    ctx.save();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;
    ctx.font = font;

    if (w > 0 && h > 0) {
      ctx.strokeRect(item.x1, item.y1, w, h);
    }

    const padding = 4;
    const textWidth = ctx.measureText(label).width;
    const textX = item.x1;
    const textY = Math.max(0, item.y1 - 18);
    ctx.fillStyle = labelBackground;
    ctx.fillRect(textX, textY, textWidth + padding * 2, 18);
    ctx.fillStyle = labelColor;
    ctx.fillText(label, textX + padding, textY + 13);

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
      overlayStateRef.current = { region: [], digit: [] };
      return;
    }

    const scaleX = clientWidth / naturalWidth;
    const scaleY = clientHeight / naturalHeight;
    const scaledRegions = buildScaledBoxes(regionBoxes, scaleX, scaleY);
    const scaledDigits = buildScaledBoxes(digitBoxes, scaleX, scaleY);

    overlayStateRef.current = {
      region: scaledRegions,
      digit: scaledDigits
    };

    drawScaledBoxes(ctx, scaledRegions, {
      strokeStyle: '#6EF2C4',
      lineWidth: 3
    });

    drawScaledBoxes(ctx, scaledDigits, {
      strokeStyle: '#F8F3CE',
      lineWidth: 1.5
    });

    if (hoveredBox?.type === 'region') {
      const item = scaledRegions[hoveredBox.index];
      drawHoverLabel(ctx, item, {
        strokeStyle: '#6EF2C4',
        lineWidth: 3,
        labelColor: '#061A12',
        labelBackground: 'rgba(110, 242, 196, 0.75)',
        labelBuilder: buildRegionLabel
      });
    }

    if (hoveredBox?.type === 'digit') {
      const item = scaledDigits[hoveredBox.index];
      drawHoverLabel(ctx, item, {
        strokeStyle: '#F8F3CE',
        lineWidth: 2,
        labelBuilder: buildDigitLabel
      });
    }
  };

  const isSameHover = (a, b) =>
    a?.type === b?.type && a?.index === b?.index;

  const findHoverTarget = (x, y) => {
    const { region, digit } = overlayStateRef.current || {};

    if (Array.isArray(digit)) {
      const digitIndex = digit.findIndex(
        (item) => x >= item.x1 && x <= item.x2 && y >= item.y1 && y <= item.y2
      );
      if (digitIndex !== -1) {
        return { type: 'digit', index: digitIndex };
      }
    }

    if (Array.isArray(region)) {
      const regionIndex = region.findIndex(
        (item) => x >= item.x1 && x <= item.x2 && y >= item.y1 && y <= item.y2
      );
      if (regionIndex !== -1) {
        return { type: 'region', index: regionIndex };
      }
    }

    return null;
  };

  const handleMouseMove = (event) => {
    const img = imageRef.current;
    if (!img) return;

    const rect = img.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      if (hoveredBox) setHoveredBox(null);
      return;
    }

    const next = findHoverTarget(x, y);
    if (!isSameHover(hoveredBox, next)) {
      setHoveredBox(next);
    }
  };

  const handleMouseLeave = () => {
    if (hoveredBox) setHoveredBox(null);
  };

  const handleImageClick = () => {
    setIsZoomed((prev) => !prev);
  };

  const csvEscape = (value) => {
    if (value === null || value === undefined) return '';
    const str = String(value);
    if (/[",\n]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const buildCsvExport = (rows) => {
    const headerRow = [
      'NRC Number',
      'Full Name',
      'Date of Birth',
      "Father's Name",
      "Mother's Name",
      'Religion',
      'Height',
      'Blood Type',
      'Blood Type Confidence',
      'Distinct Feature',
      'Issue Date',
      'Expiry Date',
      'Overall Confidence'
    ];
    const header = headerRow.map(csvEscape).join(',');
    const body = rows.map((row) => row.map(csvEscape).join(',')).join('\n');
    return `${header}\n${body}`;
  };

  const buildCsvRow = (data) => ([
    nrcNumberDisplay || '',
    data.name || '',
    data.birthDate || '',
    data.fatherName || '',
    data.motherName || '',
    data.religion || '',
    data.height || '',
    data.bloodType || '',
    typeof data.bloodTypeConfidence === 'number' ? (data.bloodTypeConfidence * 100).toFixed(0) + '%' : '',
    data.distinctFeature || data.feature || data.address || '',
    data.issueDate || '',
    data.expiryDate || '',
    typeof data.confidence === 'number' ? (data.confidence * 100).toFixed(0) + '%' : ''
  ]);

  const loadSavedRows = () => {
    try {
      const raw = localStorage.getItem('nrcScanCsvRows');
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };

  const saveRows = (rows) => {
    try {
      localStorage.setItem('nrcScanCsvRows', JSON.stringify(rows));
    } catch {
      // ignore storage failures
    }
  };

  const handleSaveInformation = () => {
    const rowData = {
      nrcNumber: nrcNumberDisplay || '',
      name: resultData.name || '',
      birthDate: resultData.birthDate || '',
      fatherName: resultData.fatherName || '',
      motherName: resultData.motherName || '',
      religion: resultData.religion || '',
      height: resultData.height || '',
      bloodType: resultData.bloodType || '',
      bloodTypeConfidence: typeof resultData.bloodTypeConfidence === 'number'
        ? (resultData.bloodTypeConfidence * 100).toFixed(0) + '%'
        : '',
      distinctFeature: resultData.distinctFeature || resultData.feature || resultData.address || '',
      issueDate: resultData.issueDate || '',
      expiryDate: resultData.expiryDate || '',
      confidence: typeof resultData.confidence === 'number'
        ? (resultData.confidence * 100).toFixed(0) + '%'
        : ''
    };

    // In Electron: append directly to one physical CSV file on disk.
    if (window?.electronAPI?.appendScannedCsv) {
      window.electronAPI.appendScannedCsv(rowData).then((res) => {
        if (!res?.ok) {
          alert(res?.error || 'Failed to save CSV');
          return;
        }
        alert(`Saved to ${res.path}`);
      }).catch((err) => {
        alert(err?.message || 'Failed to save CSV');
      });
      return;
    }

    // Browser fallback: re-download merged CSV from local storage rows.
    const rows = loadSavedRows();
    rows.push(buildCsvRow(resultData));
    saveRows(rows);
    const text = buildCsvExport(rows);
    const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'nrc-scan.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    if (!hasImage) return;
    const handleResize = () => drawOverlay();
    window.addEventListener('resize', handleResize);
    drawOverlay();
    return () => window.removeEventListener('resize', handleResize);
  }, [hasImage, scannedImage, scannedData, hoveredBox, isZoomed]);

  useEffect(() => {
    setHoveredBox(null);
  }, [scannedData, scannedImage]);

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
                <div
                  className={`result-image-wrapper ${isZoomed ? 'zoomed' : ''}`}
                  onClick={handleImageClick}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={handleMouseLeave}
                >
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
                  <div className="info-label">Issue Date</div>
                  <div className="info-value">{resultData.issueDate}</div>
                </div>
                <div className="info-item">
                  <div className="info-label">Full Name</div>
                  <div className="info-value">{resultData.name}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Father's Name</div>
                  <div className="info-value">{resultData.fatherName}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Date of birth</div>
                  <div className="info-value">{resultData.birthDate}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Religion</div>
                  <div className="info-value">{resultData.religion || '-'}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Height</div>
                  <div className="info-value">{resultData.height || '-'}</div>
                </div>

                <div className="info-item">
                  <div className="info-label">Bloodtype</div>
                  <div className="info-value">{resultData.bloodType || '-'}</div>
                  {typeof resultData.bloodTypeConfidence === 'number' && resultData.bloodTypeConfidence > 0 && (
                    <div className="info-subvalue">
                      Confidence: {Math.round(resultData.bloodTypeConfidence * 100)}%
                    </div>
                  )}
                </div>
              </div>

              {/* Distinct Feature - Full width */}
              <div className="info-item full-width">
                <div className="info-label">Distinct Feature</div>
                <div className="info-value">
                  {resultData.distinctFeature || resultData.feature || resultData.address || '-'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="result-actions">
          <button className="primary-btn" onClick={handleSaveInformation}>
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
