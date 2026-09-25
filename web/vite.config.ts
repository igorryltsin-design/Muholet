import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'

// Версия в шапке: пакет + короткий хеш коммита — по хешу сразу видно, что образ
// собран из другой ревизии. В докер-сборке .git недоступен (исключён контекстом),
// хеш туда передаётся build-arg'ом GIT_SHA (см. Makefile::docker-image)
function appVersion(): string {
  const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')) as { version: string }
  let sha = process.env.GIT_SHA ?? ''
  if (!sha) {
    try {
      sha = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()
    } catch {
      // вне git-репозитория — достаточно версии пакета
    }
  }
  return `v${pkg.version}${sha ? ` · ${sha}` : ''}`
}

export default defineConfig({
  // для статики GitHub Pages сайт живёт по пути /<репо>/: сборка с VITE_BASE
  base: process.env.VITE_BASE ?? '/',
  plugins: [react()],
  define: { __APP_VERSION__: JSON.stringify(appVersion()) },
  server: {
    port: 5173,
    proxy: {
      // порт стенда можно переопределить для параллельного инстанса: VITE_API_PORT=8092 npm run dev
      '/api': { target: `http://127.0.0.1:${process.env.VITE_API_PORT ?? 8091}`, ws: true },
    },
  },
})
