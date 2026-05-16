"""Auto-updater for PylaAI.

Checks a GitHub repository for newer commits and downloads updated files
while preserving the user's local configuration (cfg/ directory).

Usage (standalone):
    python tools/auto_updater.py              # check & apply updates
    python tools/auto_updater.py --check      # only check, don't apply

Integrated into main.py startup when ``auto_update = "yes"`` is set in
``cfg/general_config.toml``.
"""

import io
import os
import shutil
import sys
import zipfile

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# File that stores the last known commit hash after a successful update.
VERSION_FILE = os.path.join(PROJECT_ROOT, ".update_version")

# Directories / files that should NEVER be overwritten by an update because
# they contain user-specific settings, caches, or runtime artefacts.
PROTECTED = {
    "cfg",
    ".update_version",
    ".venv",
    "venv",
    "debug_frames",
    "latest_brawler_data.json",
    "models/.ort_cache",
}


def _load_toml_config():
    """Read general_config.toml without importing utils (avoid circular deps)."""
    try:
        import toml
        path = os.path.join(PROJECT_ROOT, "cfg", "general_config.toml")
        if os.path.isfile(path):
            with open(path, "r") as f:
                return toml.load(f)
    except Exception:
        pass
    return {}


def _repo_info():
    """Return ``(owner, repo, branch)`` from config or defaults."""
    cfg = _load_toml_config()
    repo = cfg.get("update_repo", "dgevvev2-hue/srcv")
    branch = cfg.get("update_branch", "main")
    parts = repo.split("/", 1)
    if len(parts) != 2:
        parts = ["dgevvev2-hue", "srcv"]
    return parts[0], parts[1], branch


def _current_commit():
    """Return the stored commit hash from the last update, or ``None``."""
    if os.path.isfile(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return f.read().strip() or None
    return None


def _save_commit(sha):
    with open(VERSION_FILE, "w") as f:
        f.write(sha)


def _is_protected(rel_path):
    """Return True if *rel_path* falls under a protected directory/file."""
    parts = rel_path.replace("\\", "/").split("/")
    for p in PROTECTED:
        p_parts = p.replace("\\", "/").split("/")
        if parts[: len(p_parts)] == p_parts:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_for_update():
    """Check GitHub for a newer commit.

    Returns ``(needs_update: bool, latest_sha: str | None, message: str)``.
    """
    owner, repo, branch = _repo_info()
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        return False, None, f"Network error checking for updates: {exc}"

    if resp.status_code != 200:
        return False, None, f"GitHub API returned {resp.status_code} when checking for updates."

    data = resp.json()
    latest_sha = data.get("sha", "")
    commit_msg = data.get("commit", {}).get("message", "").split("\n")[0]
    local_sha = _current_commit()

    if local_sha and local_sha == latest_sha:
        return False, latest_sha, "Already up to date."

    if local_sha:
        msg = f"Update available: {latest_sha[:8]} - {commit_msg}"
    else:
        msg = f"Latest remote commit: {latest_sha[:8]} - {commit_msg}"
    return True, latest_sha, msg


def apply_update(latest_sha=None):
    """Download and apply the latest version from GitHub.

    Preserves everything under ``cfg/`` and other protected paths so user
    settings are never lost.

    Returns ``(success: bool, message: str)``.
    """
    owner, repo, branch = _repo_info()

    if latest_sha is None:
        needs, latest_sha, msg = check_for_update()
        if not needs:
            return True, msg
        if latest_sha is None:
            return False, msg

    zip_url = f"https://github.com/{owner}/{repo}/archive/{latest_sha}.zip"
    print(f"Downloading update from {zip_url} ...")
    try:
        resp = requests.get(zip_url, timeout=120)
    except requests.RequestException as exc:
        return False, f"Failed to download update: {exc}"

    if resp.status_code != 200:
        return False, f"GitHub returned {resp.status_code} when downloading the update."

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except zipfile.BadZipFile:
        return False, "Downloaded file is not a valid zip archive."

    # The zip contains a single top-level directory like "srcv-<sha>/".
    top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
    if len(top_dirs) != 1:
        return False, "Unexpected zip structure."
    prefix = top_dirs.pop() + "/"

    updated = 0
    skipped = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        rel = info.filename[len(prefix):]
        if not rel:
            continue
        if _is_protected(rel):
            skipped += 1
            continue
        dest = os.path.join(PROJECT_ROOT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(info) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
        updated += 1

    _save_commit(latest_sha)
    return True, f"Updated {updated} files (skipped {skipped} protected). Version: {latest_sha[:8]}"


def auto_update_on_startup():
    """Called from main.py at startup when auto_update is enabled.

    Prints status messages and applies the update if one is available.
    Returns True if an update was applied (caller may want to restart).
    """
    print("Checking for updates...")
    needs, sha, msg = check_for_update()
    print(msg)
    if not needs:
        return False
    ok, result_msg = apply_update(sha)
    print(result_msg)
    if ok and needs:
        print(
            "Update applied. Restart the bot to use the new version.\n"
            "  (Set auto_update = \"no\" in cfg/general_config.toml to disable.)"
        )
    return ok and needs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_only = "--check" in sys.argv
    if check_only:
        _, sha, msg = check_for_update()
        print(msg)
    else:
        auto_update_on_startup()
