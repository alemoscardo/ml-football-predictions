import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative base: the built site works from any path, e.g. GitHub Pages' /<repo>/.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
