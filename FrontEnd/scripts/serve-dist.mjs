import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(scriptFile);
const projectRoot = path.resolve(scriptDir, "..");
const distRoot = path.join(projectRoot, "dist");
const port = Number.parseInt(process.env.PORT ?? "4173", 10);

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".webp", "image/webp"],
  [".gif", "image/gif"],
  [".ico", "image/x-icon"],
  [".woff", "font/woff"],
  [".woff2", "font/woff2"],
  [".ttf", "font/ttf"],
]);

if (!existsSync(distRoot)) {
  console.error(`[error] dist folder not found: ${distRoot}`);
  process.exit(1);
}

function resolveRequestPath(urlPath) {
  const cleaned = decodeURIComponent((urlPath ?? "/").split("?")[0]).replace(/^\/+/, "");
  const relativePath = cleaned === "" ? "index.html" : cleaned;
  const resolvedPath = path.resolve(distRoot, relativePath);

  if (!resolvedPath.startsWith(distRoot)) {
    return null;
  }

  if (existsSync(resolvedPath) && statSync(resolvedPath).isFile()) {
    return resolvedPath;
  }

  if (!path.extname(relativePath)) {
    return path.join(distRoot, "index.html");
  }

  return undefined;
}

createServer((req, res) => {
  const resolvedPath = resolveRequestPath(req.url);

  if (resolvedPath === null) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }

  if (!resolvedPath) {
    res.statusCode = 404;
    res.end("Not Found");
    return;
  }

  const extension = path.extname(resolvedPath).toLowerCase();
  res.setHeader("Content-Type", contentTypes.get(extension) ?? "application/octet-stream");
  createReadStream(resolvedPath).pipe(res);
}).listen(port, () => {
  console.log(`[serve:dist] Serving ${distRoot}`);
  console.log(`[serve:dist] Open http://localhost:${port}`);
});
