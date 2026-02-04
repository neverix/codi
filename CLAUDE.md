# Claude setup
> claude/ is where you'll live and store tasks. In claude/experiments/<name>_<date>, create individual experiments I ask you to perform. Store the configurations, keep track of experiment results and status.
> Model: Claude Opus 4.5

`claude/` lives in a separate directory so the main codebase is human-understandable and clean. The `claude/` code can change quickly, but it must follow the principles outlined in this document.

## Working Guidelines
- **Flag issues proactively**: Note problems, bugs, code smells in this file or task files
- **Mark uncertainty**: Many observations are low-confidence; say so explicitly
- **Ask rather than assume**: When unsure if code is used/important, ask
- **Be laconic**: In communication and code, be laconic and to-the-point.
- **Use bash find, not Glob tool**: The Glob tool is slow and sometimes rejected. Use `find <dir> -name "*.py" -type f` via Bash instead.
- **No CUDA without SLURM**: Never run GPU code directly. Always submit via SLURM.
- **No direct compute node access**: Can't run commands on compute nodes from login node. Use `srun` or submit debug jobs to run commands on compute nodes (e.g., check caches, fix symlinks).
- **Node HF cache**: Fixed 2026-01-26. `setup-node.sh` auto-removes stale local HF caches and replaces with symlink. **2026-01-28**: Fixed self-referential symlink at `$VAST/.cache/huggingface` - the TARGET dir must be a real directory, not a symlink.
- **Credentials**: Wandb/HF auth handled by `/workspace-vast/sshlin/afel-inst/setup-node.sh`, called via `claude/scripts/setup.sh`. Creates per-node symlinks (`~/.netrc`, `~/.cache/huggingface`) pointing to shared storage. Marker is per-node: `.setup.done.$(hostname)`.
- **Analysis scripts**: Always use `.venv/bin/python` for scripts needing matplotlib/numpy/etc. System python doesn't have these.
- **Gitignores**: Keep gitignores local (in the directory where ignored files live) and general (use patterns like `*.csv` rather than specific filenames). Submodules have their own `.gitignore`.
- **Async messages**: User messages arrive when Claude finishes current action, not immediately. User can't interrupt mid-action (escape stops but doesn't deliver message). Plan accordingly.
- **sacct unreliable**: `sacct` often fails with "Connection refused". Use `squeue` for running jobs, check log files directly for completed jobs.
- **Never inline setup-node.sh**: If a job gets stuck waiting for setup-node.sh lock, investigate what's holding the lock (check other jobs). Don't try to bypass by inlining setup logic - that's a workaround that ignores the real problem.
- **Never set HOME=$VAST**: Don't override HOME to work around compute node issues. This is a terrible hack that breaks many tools. Fix the actual symlink/setup issues instead.
- **setup-node.sh lock is intentionally shared**: The lock at `$INST/.setup.lock` is shared across all nodes ON PURPOSE - parallel setup runs would corrupt the VAST folder. **2026-01-28 FIXES**: (1) Lock FD inheritance - added `exec {lock_fd}>&-` to close after setup. (2) Logic bug - after acquiring blocking lock, must check if THIS node's done marker exists (not just any node). (3) Use `exit` not `return` since script is called with `bash` not `source`.
