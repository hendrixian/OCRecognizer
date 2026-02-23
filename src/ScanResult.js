import React from 'react';
import './ScanResult.css';

const ScanResult = ({ onBack, onNewScan, scannedData }) => {
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
