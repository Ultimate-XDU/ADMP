import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8787",
        changeOrigin: true,
        timeout: 600_000,      // 10 分钟，索引大文档需要加载模型+生成向量
        proxyTimeout: 600_000,
        ws: false,
      },
    },
  },
});
