import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ensureDependencies } from "./bootstrap.mjs";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(scriptFile);
const projectRoot = path.resolve(scriptDir, "..");

function runNodeScript(scriptPath, args = []) {
  const result = spawnSync(process.execPath, [scriptPath, ...args], {
    cwd: projectRoot,
    stdio: "inherit",
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

ensureDependencies();

runNodeScript(path.join(projectRoot, "node_modules", "typescript", "bin", "tsc"), ["-b"]);
runNodeScript(path.join(projectRoot, "node_modules", "vite", "bin", "vite.js"), ["build"]);
