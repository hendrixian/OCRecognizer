import React, { useState } from 'react';
import './App.css';
import NrcScanner from './NrcScannerHome';
import PhotoUpload from './PhotoUpload';
import ScanResult from './ScanResult';
import Squares from './Squares';

function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home' or 'photo'
  const [scannedData, setScannedData] = useState(null);
  const handlePhotoClick = () => {
    setCurrentPage('photo');
  };

  const handleBackToHome = () => {
    setCurrentPage('home');
  };

  const handleScanComplete = (data) => {
    setScannedData(data);
    setCurrentPage('result');
  };
  const handleNewScan = () => {
    setScannedData(null);
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
          <NrcScanner onPhotoClick={handlePhotoClick} />
        ) : (
          <PhotoUpload onBack={handleBackToHome}
                       onScanComplete={handleScanComplete} />
        )}
        {currentPage === 'result' &&(
          <ScanResult 
             onBack={handleBackToHome}
             onNewScan={handleNewScan}
             scannedData={scannedData}
            />
        )}
      </div>
    </div>
  );
}

export default App;