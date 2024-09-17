Invoke-WebRequest -Uri https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.2.0%2Bcu121.zip -OutFile "libtorch.zip"
Expand-Archive -Path "libtorch.zip" -DestinationPath "."
Remove-Item -Path "libtorch.zip"

$directory = ".cargo"
if (-Not (Test-Path -Path $directory)) {
    New-Item -ItemType Directory -Force -Path $directory
}

$currentPath = (Get-Location).Path -replace '\\', '/'

$content = @"
[env]
LIBTORCH = "$currentPath/libtorch/"
Path = "$PATH;$currentPath/libtorch/"
"@

Set-Content -Path ".cargo/config.toml" -Value $content
