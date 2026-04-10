import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptFile = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(scriptFile);
const projectRoot = path.resolve(scriptDir, "..");
const stampPath = path.join(projectRoot, ".platform-stamp.json");
const packageLockPath = path.join(projectRoot, "package-lock.json");
const nodeModulesPath = path.join(projectRoot, "node_modules");

function readStamp() {
  if (!existsSync(stampPath)) return null;

  try {
    return JSON.parse(readFileSync(stampPath, "utf8"));
  } catch {
    return null;
  }
}

function getCurrentStamp() {
  return {
    platform: process.platform,
    arch: process.arch,
    nodeMajor: Number.parseInt(process.versions.node.split(".")[0] ?? "0", 10),
  };
}

function isPackageLockNewerThanStamp() {
  if (!existsSync(packageLockPath) || !existsSync(stampPath)) return false;
  return statSync(packageLockPath).mtimeMs > statSync(stampPath).mtimeMs;
}

function shouldInstall() {
  if (!existsSync(nodeModulesPath)) return true;

  const saved = readStamp();
  const current = getCurrentStamp();

  if (!saved) return true;
  if (saved.platform !== current.platform) return true;
  if (saved.arch !== current.arch) return true;
  if (saved.nodeMajor !== current.nodeMajor) return true;
  if (isPackageLockNewerThanStamp()) return true;

  return false;
}

function runNpmInstall() {
  const result = spawnSync("npm install", {
    cwd: projectRoot,
    shell: true,
    stdio: "inherit",
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

export function ensureDependencies() {
  if (shouldInstall()) {
    console.log(`[setup] Syncing dependencies for ${process.platform}-${process.arch}...`);
    runNpmInstall();
  }

  writeFileSync(stampPath, `${JSON.stringify(getCurrentStamp(), null, 2)}\n`, "utf8");
}

if (process.argv[1] === scriptFile) {
  ensureDependencies();
}
