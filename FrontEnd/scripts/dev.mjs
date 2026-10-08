import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ensureDependencies } from "./bootstrap.mjs";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(scriptFile);
const projectRoot = path.resolve(scriptDir, "..");

ensureDependencies();

const viteBin = path.join(projectRoot, "node_modules", "vite", "bin", "vite.js");
const child = spawn(process.execPath, [viteBin, ...process.argv.slice(2)], {
  cwd: projectRoot,
  stdio: "inherit",
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
