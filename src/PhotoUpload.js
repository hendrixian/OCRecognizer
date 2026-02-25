import React, { useState, useRef } from 'react';
import './PhotoUpload.css';
import { scanNrcFromDataUrl } from './services/ocrService';
import { toUiScanResult } from './services/scanResultAdapter';

const PhotoUpload = ({ onBack, onScanComplete }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFile(file);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      handleFile(file);
    }
  };

  const handleFile = (file) => {
    // Check if file is an image
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please select an image file (JPG or PNG)');
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleBrowseClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    triggerFileInput();
  };

  const removeImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle scan button click
  const handleScan = async () => {
    if (!selectedImage || isScanning) return;

    setIsScanning(true);

    try {
      const apiResult = await scanNrcFromDataUrl(selectedImage);
      const uiResult = toUiScanResult(apiResult);
      onScanComplete(uiResult, selectedImage);
    } catch (err) {
      console.error(err);
      alert(err?.message || 'Scan failed. Please try again.');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="photo-upload-container">
      <div className="upload-card">
        {/* Header with back button and title */}
        <div className="upload-header">
          <button className="back-button" onClick={onBack}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="#F8F3CE" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h1 className="upload-title">NRC SCANNER</h1>
          <div className="title-underline"></div>
        </div>

        {/* File format indicator */}
        <div className="file-format-badge">
          <span className="file-format">File Format: JPG, PNG</span>
        </div>

        {/* Instruction text */}
        <p className="upload-instruction">
          Please drag or input your photo here
        </p>

        {/* Upload area */}
        <div
          className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedImage ? 'has-image' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={selectedImage ? null : triggerFileInput}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          {selectedImage ? (
            <div className="image-preview-container">
              <img src={selectedImage} alt="Preview" className="image-preview" />
              <button className="remove-image-btn" onClick={removeImage}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M18 6L6 18M6 6L18 18" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </button>
            </div>
          ) : (
            <div className="upload-placeholder">
              <div className="upload-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="2" y="5" width="20" height="16" rx="2" stroke="#F8F3CE" strokeWidth="1.5"/>
                  <circle cx="17" cy="10" r="2" stroke="#F8F3CE" strokeWidth="1.5"/>
                  <path d="M4 18L8 12L12 15L16 10L20 15" stroke="#F8F3CE" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M16 5L19 2" stroke="#F8F3CE" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <p className="drag-text">Drag & drop your image here</p>
              <p className="or-text">or</p>
              <button type="button" className="browse-btn" onClick={handleBrowseClick}>
                Browse Files
              </button>
            </div>
          )}
        </div>

        {/* Action buttons */}
        {selectedImage && (
          <div className="action-buttons">
            <button 
              className="scan-btn" 
              onClick={handleScan}
              disabled={isScanning}
            >
              {isScanning ? 'Scanning...' : 'Scan Document'}
            </button>
            <button className="change-btn" onClick={removeImage}>
              Choose Different
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhotoUpload;
