Invoke-WebRequest -Uri https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.2.0%2Bcu121.zip -OutFile "libtorch.zip"
Expand-Archive -Path "libtorch.zip" -DestinationPath "."
Remove-Item -Path "libtorch.zip"
