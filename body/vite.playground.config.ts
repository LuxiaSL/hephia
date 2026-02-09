import { resolve } from 'path';
import { defineConfig } from 'vite';

// Standalone playground config — no Svelte, no Tauri, just WebGL
export default defineConfig({
  resolve: {
    alias: {
      '$creature': resolve(__dirname, 'src/creature'),
      '$state': resolve(__dirname, 'src/state'),
      '$lib': resolve(__dirname, 'src/lib'),
    },
  },
  root: resolve(__dirname, 'playground'),
  server: {
    port: 1421,
  },
});
