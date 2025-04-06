@echo off
setlocal

:: Define installation paths
set ANACONDA_INSTALLER=Anaconda3-latest-Windows-x86_64.exe
set ANACONDA_URL=https://repo.anaconda.com/archive/%ANACONDA_INSTALLER%
set VS_INSTALLER=vs_community.exe
set VS_URL=https://aka.ms/vs/17/release/vs_Community.exe

:: Check if Anaconda is installed
where conda > nul 2>&1
if %errorlevel% neq 0 (
    echo Downloading Anaconda...
    curl -o %ANACONDA_INSTALLER% %ANACONDA_URL%
    echo Installing Anaconda...
    start /wait %ANACONDA_INSTALLER% /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S
) else (
    echo Anaconda is already installed. Skipping...
)

:: Check if Visual Studio is installed
where cl > nul 2>&1
if %errorlevel% neq 0 (
    echo Downloading Visual Studio Installer...
    curl -o %VS_INSTALLER% %VS_URL%
    echo Installing Visual Studio with C++ and .NET...
    start /wait %VS_INSTALLER% --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.NetCore.Component.Runtime --quiet --norestart
) else (
    echo Visual Studio is already installed. Skipping...
)

:: Ensure Anaconda is in the path
call conda --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Conda is not recognized. Adding to PATH...
    setx PATH "%USERPROFILE%\Anaconda3\Scripts;%USERPROFILE%\Anaconda3\Library\bin;%PATH%"
)

:: Create Anaconda environment if not exists
call conda info --envs | findstr /C:"myenv" > nul
if %errorlevel% neq 0 (
    echo Creating Anaconda environment...
    call conda create -n myenv python=3.10 -y
) else (
    echo Anaconda environment 'myenv' already exists. Skipping...
)

:: Activate environment
call conda activate myenv

:: Check for CUDA availability
echo Checking for CUDA support...
python -c "import torch; print(torch.cuda.is_available())" > check_gpu.txt
findstr "True" check_gpu.txt > nul
if %errorlevel% == 0 (
    echo Installing PyTorch with CUDA...
    call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
) else (
    echo Installing CPU-only PyTorch...
    call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
)

:: Install additional Python packages
echo Installing required libraries...
call pip install datasets transformers numpy pandas

echo Setup Complete!
pause
