import { defineConfig } from 'vite'

// Keep browser requests relative to the UI origin during development, while
// forwarding API calls to the FastAPI server.
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
