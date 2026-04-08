import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@sim": path.resolve(__dirname, "../1dof_pinn/sim"),
      "@": path.resolve(__dirname, "."),
    },
  },
});
