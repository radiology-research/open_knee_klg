# open_knee_klg

# Overview
This project contains the code and models for detecting knee joints from standing AP radiographs and grading them for Kellgren-Lawrence grades (KLG) for assessing radiographic knee osteoarthritis (rkOA)

## Quick start

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Manually run prediction on DICOM file
uv run run.py -i /path/to/input/knee_xr.dcm -o /path/to/output
```