#!/usr/bin/env python3
# Usage: python3 ~/mark_fixed.py <POTHOLE_ID>
import sys, os, subprocess, shlex

OUT_DIR = os.path.expanduser('~/detections')
FIXED = os.path.join(OUT_DIR, 'fixed_ids.txt')

if len(sys.argv) != 2:
    print("Usage: mark_fixed.py <pothole_id>"); sys.exit(1)

pid = sys.argv[1].strip()
os.makedirs(OUT_DIR, exist_ok=True)
# append if not present
if os.path.exists(FIXED):
    existing = open(FIXED).read().splitlines()
else:
    existing = []
if pid not in existing:
    with open(FIXED, 'a') as f: f.write(pid+"\n")
    print("Added", pid, "to fixed_ids.txt")
else:
    print(pid, "already in fixed_ids.txt")

# rebuild geojsons
cmd = "python3 " + shlex.quote(os.path.expanduser('~/aggregate_potholes.py'))
subprocess.run(cmd, shell=True, check=False)
