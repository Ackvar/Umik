param(
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".runtime-venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$MainScript = Join-Path $ProjectRoot "main.py"
$MinPythonMajor = 3
$MinPythonMinor = 11

function Write-Step {
    param([string]$Message)
    Write-Host "[BOOT] $Message"
}

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-UsablePython {
    $candidates = @()

    if (Test-Path $PythonExe) {
        $candidates += $PythonExe
    }

    $localPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
    if (Test-Path $localPython) {
        $candidates += $localPython
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $candidates += $pyCmd.Source
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -notlike "*WindowsApps*") {
        $candidates += $pythonCmd.Source
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        try {
            if ($candidate -like "*\py.exe") {
                $launcherArg = "-$MinPythonMajor.$MinPythonMinor"
                $versionText = & $candidate $launcherArg -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
                if ($LASTEXITCODE -eq 0) {
                    return @{
                        Path = $candidate
                        Args = @($launcherArg)
                    }
                }
                continue
            }

            $versionText = & $candidate -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
            if ($LASTEXITCODE -ne 0) {
                continue
            }

            $parts = $versionText.Trim().Split(".")
            if ($parts.Length -lt 2) {
                continue
            }

            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -gt $MinPythonMajor -or ($major -eq $MinPythonMajor -and $minor -ge $MinPythonMinor)) {
                return @{
                    Path = $candidate
                    Args = @()
                }
            }
        } catch {
            continue
        }
    }

    return $null
}

function Install-PythonIfMissing {
    $pythonInfo = Get-UsablePython
    if ($pythonInfo) {
        return $pythonInfo
    }

    if (-not (Test-CommandExists "winget")) {
        throw "Python $MinPythonMajor.$MinPythonMinor+ не найден, а winget недоступен для автоустановки."
    }

    Write-Step "Python $MinPythonMajor.$MinPythonMinor+ не найден. Устанавливаю через winget..."
    winget install --id Python.Python.3.13 -e --silent --accept-package-agreements --accept-source-agreements

    $pythonInfo = Get-UsablePython
    if (-not $pythonInfo) {
        throw "Python установлен, но не найден автоматически. Перезапустите терминал и повторите запуск."
    }

    return $pythonInfo
}

function Ensure-Venv {
    param([hashtable]$PythonInfo)

    if (Test-Path $PythonExe) {
        Write-Step "Использую существующее окружение $VenvPath"
        return
    }

    Write-Step "Создаю виртуальное окружение $VenvPath"
    & $PythonInfo.Path @($PythonInfo.Args + @("-m", "venv", $VenvPath))
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExe)) {
        throw "Не удалось создать виртуальное окружение."
    }
}

function Ensure-Pip {
    Write-Step "Проверяю pip"
    & $PythonExe -m ensurepip --upgrade | Out-Null
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось подготовить pip."
    }
}

function Ensure-Dependencies {
    if (-not (Test-Path $RequirementsPath)) {
        throw "Файл requirements.txt не найден: $RequirementsPath"
    }

    Write-Step "Устанавливаю Python-зависимости из requirements.txt"
    & $PythonExe -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось установить Python-зависимости."
    }
}

function Ensure-Ffmpeg {
    if (Test-CommandExists "ffmpeg") {
        Write-Step "ffmpeg уже доступен"
        return
    }

    if (-not (Test-CommandExists "winget")) {
        Write-Step "ffmpeg не найден и winget недоступен. AAC-сохранение может не работать."
        return
    }

    Write-Step "ffmpeg не найден. Устанавливаю через winget..."
    winget install --id Gyan.FFmpeg -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Не удалось автоматически установить ffmpeg. AAC-сохранение может не работать."
    } else {
        Write-Step "ffmpeg установлен. Если текущая сессия не видит PATH, перезапустите терминал позже."
    }
}

Push-Location $ProjectRoot
try {
    Write-Step "Проверяю Python"
    $pythonInfo = Install-PythonIfMissing

    Ensure-Venv -PythonInfo $pythonInfo
    Ensure-Pip
    Ensure-Dependencies
    Ensure-Ffmpeg

    if ($SkipRun) {
        Write-Step "Проверка завершена, запуск проекта пропущен"
    } else {
        Write-Step "Запускаю проект"
        & $PythonExe $MainScript
    }
} finally {
    Pop-Location
}
