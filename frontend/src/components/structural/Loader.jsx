// Loader.js
import React from 'react';
import { ScaleLoader } from 'react-spinners'



const Loader = () => {
  const loaderStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    backgroundColor: 'rgba(0,0,0,0.25)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999,
  };

  return (
    <div style={loaderStyle}>
      <ScaleLoader color="#3a72ffff" height="167" width="20" speedMultiplier="0.67" radius="15" animation="border" variant="light" />
    </div>
  );
};

export default Loader;
