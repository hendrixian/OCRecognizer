import React, { useState } from 'react';
import './App.css';
import NrcScanner from './NrcScannerHome';
import PhotoUpload from './PhotoUpload';
import CameraScan from './CameraScan';
import ScanResult from './ScanResult';
import Squares from './Squares';

function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home' | 'photo' | 'camera' | 'result'
  const [scannedData, setScannedData] = useState(null);
  const [scannedImage, setScannedImage] = useState(null);

  const handlePhotoClick = () => {
    setCurrentPage('photo');
  };
  const handleCameraClick = () => {
    setCurrentPage('camera');
  };

  const handleBackToHome = () => {
    setScannedData(null);
    setScannedImage(null);
    setCurrentPage('home');
  };

  const handleScanComplete = (data, imageDataUrl) => {
    setScannedData(data);
    setScannedImage(imageDataUrl || null);
    setCurrentPage('result');
  };

  const handleNewScan = () => {
    setScannedData(null);
    setScannedImage(null);
    setCurrentPage('photo');
  };

  return (
    <div className="App">
      <Squares 
        speed={0.25}
        squareSize={40}
        direction="diagonal"
        borderColor="#000000"
        hoverFillColor="#2e4538"
      />
      <div className="content-wrapper">
        {currentPage === 'home' ? (
          <NrcScanner onPhotoClick={handlePhotoClick} onCameraClick={handleCameraClick} />
        ) : null}
        {currentPage === 'photo' && (
          <PhotoUpload
            onBack={handleBackToHome}
            onScanComplete={handleScanComplete}
          />
        )}
        {currentPage === 'camera' && (
          <CameraScan
            onBack={handleBackToHome}
            onScanComplete={handleScanComplete}
          />
        )}
        {currentPage === 'result' && (
          <ScanResult
            onBack={handleBackToHome}
            onNewScan={handleNewScan}
            scannedData={scannedData}
            scannedImage={scannedImage}
          />
        )}
      </div>
    </div>
  );
}

export default App;
