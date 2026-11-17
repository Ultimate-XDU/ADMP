import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";

const project = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(resolve(project, "package.json"));
const { build } = require("esbuild");
const cache = resolve(project, "node_modules/.cache");
mkdirSync(cache, { recursive: true });
const bundle = resolve(cache, "admp-citations.cjs");

await build({
  entryPoints: [resolve(project, "tests/citations.tsx")],
  bundle: true,
  platform: "node",
  format: "cjs",
  jsx: "automatic",
  outfile: bundle,
  logLevel: "warning",
});

const result = require(bundle).runChecks();
const source = resolve(project, "src/components/AnswerText.tsx");
console.log(
  JSON.stringify(
    {
      ...result,
      source,
      sourceSha256: createHash("sha256")
        .update(readFileSync(source))
        .digest("hex"),
      validation:
        "Real React rendering and citation button handlers. No backend mocks.",
    },
    null,
    2,
  ),
);
