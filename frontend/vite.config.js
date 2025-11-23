import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Replace 'username' and 'repo' with your GitHub info
const repoName = 'MadHacksReal'

export default defineConfig({
  plugins: [react()],
  base: `/`, // required for GitHub Pages
  build: {
    outDir: '../docs',  // build goes to root/docs
    emptyOutDir: true,  // clears old build files
  },
})

