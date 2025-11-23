import { useState, useEffect } from 'react'
import { HashRouter, Route, Routes } from 'react-router';

import './App.css'
import Layout from './components/Layout';
import LandingPage from './components/pages/LandingPage';
import NoMatch from './components/pages/NoMatch';



function App() {

  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<LandingPage />} />
          <Route path="*" element={<NoMatch />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}

export default App
