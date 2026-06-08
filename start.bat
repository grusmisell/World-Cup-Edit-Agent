@echo off
REM World Cup Edit Agent launcher for Windows
cd /d "%~dp0"

if not exist ".env" (
  echo No .env found. Copying .env.example to .env ...
  copy ".env.example" ".env" >nul
  echo.
  echo  IMPORTANT: open .env and paste your ANTHROPIC_API_KEY, then run start.bat again.
  echo.
  pause
  exit /b
)

python run.py
pause
