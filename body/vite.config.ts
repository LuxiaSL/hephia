import { resolve } from 'path';
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      '$lib': resolve(__dirname, 'src/lib'),
      '$creature': resolve(__dirname, 'src/creature'),
      '$state': resolve(__dirname, 'src/state'),
    },
  },
  build: {
    rollupOptions: {
      input: {
        overlay: resolve(__dirname, 'src/overlay/index.html'),
        chat: resolve(__dirname, 'src/chat/index.html'),
        dashboard: resolve(__dirname, 'src/dashboard/index.html'),
        wizard: resolve(__dirname, 'src/wizard/index.html'),
      },
    },
  },
  // Tauri expects a fixed port during dev
  server: {
    port: 1420,
    strictPort: true,
  },
});
