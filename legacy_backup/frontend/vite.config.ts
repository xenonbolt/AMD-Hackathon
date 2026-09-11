import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
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

  const localStoragePlugin = {
    name: 'localstorage-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/local-api/history' && req.method === 'GET') {
          try {
            const historyPath = path.resolve(__dirname, 'localstorage', 'history.json');
            if (fs.existsSync(historyPath)) {
              const data = fs.readFileSync(historyPath, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(data);
            } else {
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify([]));
            }
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        } else if (req.url === '/local-api/history' && req.method === 'POST') {
          let body = '';
          req.on('data', chunk => {
            body += chunk.toString();
          });
          req.on('end', () => {
            try {
              const dirPath = path.resolve(__dirname, 'localstorage');
              if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
              }
              const historyPath = path.resolve(dirPath, 'history.json');
              fs.writeFileSync(historyPath, body, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ success: true }));
            } catch (e) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: String(e) }));
            }
          });
        } else if (req.url === '/local-api/session' && req.method === 'GET') {
          try {
            const sessionPath = path.resolve(__dirname, 'localstorage', 'session.json');
            if (fs.existsSync(sessionPath)) {
              const data = fs.readFileSync(sessionPath, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(data);
            } else {
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify(null));
            }
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        } else if (req.url === '/local-api/session' && req.method === 'POST') {
          let body = '';
          req.on('data', chunk => {
            body += chunk.toString();
          });
          req.on('end', () => {
            try {
              const dirPath = path.resolve(__dirname, 'localstorage');
              if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
              }
              const sessionPath = path.resolve(dirPath, 'session.json');
              fs.writeFileSync(sessionPath, body, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ success: true }));
            } catch (e) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: String(e) }));
            }
          });
        } else {
          next();
        }
      });
    }
  };

  return {
    plugins: [react(), tailwindcss(), localStoragePlugin],
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
      watch: process.env.DISABLE_HMR === 'true' ? null : {
        ignored: ['**/localstorage/**']
      },
      proxy: proxyConfig
    },
  };
});
