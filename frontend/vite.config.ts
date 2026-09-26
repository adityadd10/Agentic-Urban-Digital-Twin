import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// dev doc §11.3: "thesis demo — keep deliberately small". The backend
// (module M10b, `udt.api.main:app`) is expected to run separately on
// port 8000 (`uv run uvicorn udt.api.main:app --app-dir src`) - this
// dev-server proxy just avoids CORS friction while developing, it is
// not needed for a production build (`VITE_API_BASE_URL` env var
// controls the real target then, see `src/api.ts`).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/episodes": "http://127.0.0.1:8000",
      "/twin": "http://127.0.0.1:8000",
      "/incidents": "http://127.0.0.1:8000",
      "/decisions": "http://127.0.0.1:8000",
      "/approvals": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
