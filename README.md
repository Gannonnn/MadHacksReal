# MadHacks

## Setting up with Conda

### Installing Conda (Windows)

If conda is not installed, you have two options:

**Option 1: Miniconda (Recommended - Lightweight)**
1. Download Miniconda for Windows from: https://docs.conda.io/en/latest/miniconda.html
2. Run the installer (`.exe` file)
3. **Important:** Check "Add Miniconda3 to my PATH environment variable" during installation
4. Restart your PowerShell terminal

**Option 2: Anaconda (Full Distribution)**
1. Download Anaconda from: https://www.anaconda.com/download
2. Run the installer
3. Check "Add Anaconda3 to my PATH environment variable" during installation
4. Restart your PowerShell terminal

**Verify Installation:**
After installation and restarting PowerShell, verify conda is available:
```powershell
conda --version
```

**If conda is still not recognized after installation:**
You may need to initialize conda for PowerShell manually:
```powershell
# Find your conda installation (usually in your user directory)
# Then run:
& "C:\Users\write\miniconda3\Scripts\conda.exe" init powershell
# Or for Anaconda:
& "C:\Users\write\anaconda3\Scripts\conda.exe" init powershell
```
Then restart PowerShell.

### Creating the Environment

1. **Create the conda environment from the environment.yml file:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate madhacks
   ```

3. **Verify the environment is active:**
   ```bash
   conda info --envs
   ```
   You should see `*` next to `madhacks` indicating it's active.
   
   **Understanding the output:**
   - `*` = currently active environment
   - `+` = frozen environment (protected from changes)

### Daily Usage

- **Activate the environment:**
  ```bash
  conda activate madhacks
  ```

- **Deactivate the environment:**
  ```bash
  conda deactivate
  ```

- **Install a new package:**
  ```bash
  conda install package-name
  # or for pip-only packages:
  pip install package-name
  ```

- **Update the environment.yml after installing packages:**
  ```bash
  conda env export > environment.yml
  ```

### Managing the Environment

- **List all environments:**
  ```bash
  conda env list
  ```

- **Remove the environment:**
  ```bash
  conda env remove -n madhacks
  ```

- **Update all packages in the environment:**
  ```bash
  conda update --all
  ```

### Troubleshooting

**Checking if your terminal is responsive:**
- Press `Enter` to see if the prompt responds
- Press `Ctrl+C` to interrupt any running command
- Type a simple command like `echo "test"` or `pwd` to verify it executes

**If conda commands don't work in a new terminal:**
You need to initialize conda for PowerShell. Run this once:
```powershell
. "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
```
Or restart PowerShell after running `conda init powershell` (which modifies your PowerShell profile).

**Frozen environments:**
If you see `+` next to an environment in `conda env list`, it means the environment is frozen (protected). To unfreeze:
```powershell
conda config --set frozen_envs ""
```

### Sharing Your Environment

The `environment.yml` file can be shared with others. They can recreate your exact environment by running:
```bash
conda env create -f environment.yml
```