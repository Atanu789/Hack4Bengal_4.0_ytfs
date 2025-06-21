import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { crx } from '@crxjs/vite-plugin';
import manifest from './manifest.json';
import tailwindcss from '@tailwindcss/vite'
import path from 'path';
import copy from 'rollup-plugin-copy'; // ✅ Correct import

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
    crx({ manifest })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      plugins: [
        copy({
          targets: [
            { src: 'icon.png', dest: 'dist' },
            { src: 'icons/*', dest: 'dist/icons' }
          ],
          hook: 'writeBundle'
        })
      ]
    }
  }
});
