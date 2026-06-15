import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  const apiBase = process.env.VITE_API_BASE || env.VITE_API_BASE;
  const token = process.env.VITE_JUPYTER_TOKEN || env.VITE_JUPYTER_TOKEN || process.env.VITE_API_TOKEN || env.VITE_API_TOKEN;

  let proxyConfig = {};
  if (apiBase) {
    const targetUrl = apiBase.replace(/\/api\/?$/, "").replace(/\/$/, "");
    proxyConfig = {
      '/api': {
        target: targetUrl,
        changeOrigin: true,
        secure: false,
        headers: token ? {
          'Authorization': `token ${token}`
        } : {},
      }
    };
  }

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      proxy: proxyConfig
    },
  };
});
