import { useState, useEffect } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router';

import './App.css'
import Layout from './components/Layout';
import LandingPage from './components/pages/LandingPage';
import NoMatch from './components/pages/NoMatch';



function App() {

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<LandingPage />} />
          <Route path="*" element={<NoMatch />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
