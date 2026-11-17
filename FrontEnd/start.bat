@echo off
setlocal

cd /d "%~dp0"
title Aviation Knowledge Command Core

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found. Please install Node.js first.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo [INFO] Installing dependencies...
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed.
    pause
    exit /b 1
  )
)

echo.
echo [INFO] Starting dev server...
echo [INFO] Browser will open automatically after startup.
echo.

call npm run dev -- --open

if errorlevel 1 (
  echo.
  echo [ERROR] Dev server exited unexpectedly.
  pause
  exit /b 1
)
