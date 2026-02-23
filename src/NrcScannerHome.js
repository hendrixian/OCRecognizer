import React, { useState } from 'react';
import './NrcScannerHome.css';

const NrcScanner = ({onPhotoClick, onCameraClick}) => {
  const [selectedFormat, setSelectedFormat] = useState(null);

  return (
    <div className="nrc-scanner-container">
      {/* Main card */}
      <div className="scanner-card">
        {/* Header with NRC SCANNER title and decorative line */}
        <div className="header">
          <h1 className="title">NRC SCANNER</h1>
          <div className="title-underline"></div>
        </div>

        {/* Instruction text */}
        <p className="instruction">
          Please choose your input format to scan
        </p>

        {/* Option cards */}
        <div className="options-grid">
          {/* Photo card */}
          <div
            className={`option-card ${selectedFormat === 'photo' ? 'selected' : ''}`}
            onClick={() => {setSelectedFormat('photo');
                           onPhotoClick();
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                setSelectedFormat('photo');
                onPhotoClick();
              }
            }}
          >
            <div className="icon-container">
              {/* Photo icon (camera + mountain landscape) */}
              <svg
                className="option-icon"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect
                  x="2"
                  y="5"
                  width="20"
                  height="16"
                  rx="2"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <circle
                  cx="17"
                  cy="10"
                  r="2"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M4 18L8 12L12 15L16 10L20 15"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M16 5L19 2"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </div>
            <span className="option-label">Photo</span>
          </div>

          {/* Camera card */}
          <div
            className={`option-card ${selectedFormat === 'camera' ? 'selected' : ''}`}
            onClick={() => {
              setSelectedFormat('camera');
              if (onCameraClick) onCameraClick();
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                setSelectedFormat('camera');
                if (onCameraClick) onCameraClick();
              }
            }}
          >
            <div className="icon-container">
              {/* Camera icon (classic camera body) */}
              <svg
                className="option-icon"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect
                  x="2"
                  y="6"
                  width="20"
                  height="14"
                  rx="2"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <circle
                  cx="12"
                  cy="13"
                  r="4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M8 4L10 2H14L16 4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
                <path
                  d="M19 10V10.5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </div>
            <span className="option-label">Camera</span>
          </div>
        </div>

        {/* Optional subtle hint */}
        <div className="footer-hint"></div>
      </div>
    </div>
  );
};

export default NrcScanner;
