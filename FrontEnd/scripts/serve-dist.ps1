param(
  [int]$Port = 4173
)

$ErrorActionPreference = "Stop"

function Get-ContentType {
  param([string]$Extension)

  switch ($Extension.ToLowerInvariant()) {
    ".html" { "text/html; charset=utf-8" }
    ".js" { "text/javascript; charset=utf-8" }
    ".mjs" { "text/javascript; charset=utf-8" }
    ".css" { "text/css; charset=utf-8" }
    ".json" { "application/json; charset=utf-8" }
    ".svg" { "image/svg+xml" }
    ".png" { "image/png" }
    ".jpg" { "image/jpeg" }
    ".jpeg" { "image/jpeg" }
    ".webp" { "image/webp" }
    ".gif" { "image/gif" }
    ".ico" { "image/x-icon" }
    ".woff" { "font/woff" }
    ".woff2" { "font/woff2" }
    ".ttf" { "font/ttf" }
    default { "application/octet-stream" }
  }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot "..\dist"))

if (-not (Test-Path -LiteralPath $distRoot)) {
  throw "dist folder not found: $distRoot"
}

$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("http://localhost:$Port/")
$listener.Start()

Write-Host "[INFO] Serving files from $distRoot"
Write-Host "[INFO] Open http://localhost:$Port"
Start-Process "http://localhost:$Port"

try {
  while ($listener.IsListening) {
    $context = $listener.GetContext()
    $requestPath = [System.Uri]::UnescapeDataString($context.Request.Url.AbsolutePath.TrimStart("/"))
    if ([string]::IsNullOrWhiteSpace($requestPath)) {
      $requestPath = "index.html"
    }

    $safeRelativePath = $requestPath.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
    $resolvedPath = [System.IO.Path]::GetFullPath((Join-Path $distRoot $safeRelativePath))

    if (-not $resolvedPath.StartsWith($distRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
      $context.Response.StatusCode = 403
      $context.Response.Close()
      continue
    }

    $extension = [System.IO.Path]::GetExtension($resolvedPath)
    $shouldFallbackToIndex = (-not $extension)

    if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
      if ($shouldFallbackToIndex) {
        $resolvedPath = Join-Path $distRoot "index.html"
      } else {
        $context.Response.StatusCode = 404
        $context.Response.Close()
        continue
      }
    }

    $bytes = [System.IO.File]::ReadAllBytes($resolvedPath)
    $context.Response.ContentType = Get-ContentType ([System.IO.Path]::GetExtension($resolvedPath))
    $context.Response.ContentLength64 = $bytes.Length
    $context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    $context.Response.OutputStream.Close()
  }
}
finally {
  if ($listener.IsListening) {
    $listener.Stop()
  }
  $listener.Close()
}
