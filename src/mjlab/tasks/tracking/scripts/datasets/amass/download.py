"""Download curated *atomic* human-motion clips from AMASS (robot-agnostic).

AMASS (Mahmood et al., ICCV 2019) is the field-standard, *robot-agnostic*
motion source: every clip is a sequence of SMPL/SMPL-X body parameters, not a
robot's joint angles.

This script grabs a couple of AMASS subsets and *curates the atomic skills* out
of them. ACCAD and KIT are the cleanest for fundamental, single-skill clips:
their files are named per motion (``B3 - walk1``, ``C3 - Run``, ``A7 - crouch``,
``jumps``, ...), so selecting walk / run / sprint / jump / crouch is a filename
match. The result is one flat ``<skill>/`` folder of SMPL-X ``.npz`` files,
ready for retargeting.

The ``kick`` skill also reaches CMU, whose files are numbered rather than named:
subjects 10 and 11 are its soccer session, six trials of running up to a ball and
kicking it. Ask for it with ``--subsets CMU --skills "('kick',)"``.

Retarget what lands here with
``mjlab.tasks.tracking.scripts.datasets.gmr.retarget``, which turns SMPL-X into
G1 joint angles on CPU.

AMASS is license-gated; there is no open mirror (the HuggingFace mirrors are all
robot-retargeted derivatives, not the agnostic source). You must:

  1. Register (free) and accept the license at https://amass.is.tue.mpg.de
  2. Pass credentials via ``--username``/``--password`` or the ``AMASS_USER`` /
     ``AMASS_PASSWORD`` environment variables.

License: AMASS is released for non-commercial research under the MPI license;
each underlying mocap subset (ACCAD, KIT, ...) keeps its own terms. Cite AMASS
(Mahmood et al., ICCV 2019) and the relevant subset in any publication.

Run with:
  uv run python src/mjlab/tasks/tracking/scripts/datasets/download.py \
      --username you@example.com --password '...'
  # only a couple skills, into a custom dir:
  uv run python src/mjlab/tasks/tracking/scripts/datasets/download.py \
      --skills walk run jump --output-dir data/amass_atomic
  # different subsets:
  uv run python src/mjlab/tasks/tracking/scripts/datasets/download.py \
      --subsets ACCAD KIT BMLrub
"""

from __future__ import annotations

import http.cookiejar
import os
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

import tyro
from tqdm import tqdm

LOGIN_URL = "https://download.is.tue.mpg.de/login.php"
DOWNLOAD_URL = "https://download.is.tue.mpg.de/download.php"

# Per-dataset SMPL-X (neutral) tarballs on the MPI download portal. If a name
# 404s, open the AMASS download page logged in, right-click the dataset's
# "SMPL-X N" link, copy the ``sfile=...`` query value, and adjust the template
# or pass the corrected name via ``--subsets``.
SFILE_TEMPLATE = "amass_per_dataset/smplx/neutral/mosh_results/{subset}.tar.bz2"

# Atomic skill -> filename keywords (matched case-insensitively against each
# clip's ``.npz`` name). Deliberately conservative: better to drop an ambiguous
# clip than to file a turning/dancing motion under "walk".
SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
  "walk": ("walk",),
  "run": ("run", "jog"),
  "sprint": ("sprint", "dash"),
  "jump": ("jump", "hop", "leap"),
  "crouch": ("crouch", "squat", "kneel"),
  "kick": ("kick", "soccer"),
}

# Skills whose clips are not named after the motion, matched on the path inside the tarball
# instead. CMU numbers its files by subject and trial, and subjects 10 and 11 are the whole
# of its soccer session: every trial there is a run up and a kick at a ball. ACCAD puts its
# martial arts kicks in a folder whose name says so while the files do not.
SKILL_PATHS: dict[str, tuple[str, ...]] = {
  "kick": ("CMU/10/", "CMU/11/11_01", "Male2MartialArtsKicks"),
}

# Subsets that carry clean, well-named atomic locomotion clips.
DEFAULT_SUBSETS: tuple[str, ...] = ("ACCAD", "KIT")


def _login(username: str, password: str) -> urllib.request.OpenerDirector:
  """Authenticate to the MPI download portal, returning a cookie-bearing opener."""
  jar = http.cookiejar.CookieJar()
  opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
  data = urllib.parse.urlencode({"username": username, "password": password}).encode()
  opener.open(LOGIN_URL, data=data).read()
  if not any(c.name in ("PHPSESSID", "is_session") for c in jar):
    raise SystemExit(
      "Login to download.is.tue.mpg.de failed. Check the credentials and that "
      "you accepted the AMASS license at https://amass.is.tue.mpg.de."
    )
  return opener


def _download_subset(
  opener: urllib.request.OpenerDirector, subset: str, dest: Path
) -> None:
  if dest.exists():
    return
  dest.parent.mkdir(parents=True, exist_ok=True)
  sfile = SFILE_TEMPLATE.format(subset=subset)
  url = f"{DOWNLOAD_URL}?{urllib.parse.urlencode({'domain': 'amass', 'sfile': sfile})}"
  tmp = dest.with_suffix(dest.suffix + ".part")
  resp = opener.open(url)
  ctype = resp.headers.get("Content-Type", "")
  if "text/html" in ctype:
    raise SystemExit(
      f"Got an HTML page instead of a tarball for '{subset}' (sfile={sfile}). "
      "The session likely expired or the sfile path is wrong -- verify it on "
      "the AMASS download page."
    )
  total = int(resp.headers.get("Content-Length", 0))
  with (
    open(tmp, "wb") as f,
    tqdm(
      total=total, unit="B", unit_scale=True, desc=f"{subset}.tar.bz2", ncols=100
    ) as bar,
  ):
    while chunk := resp.read(1 << 16):
      f.write(chunk)
      bar.update(len(chunk))
  tmp.replace(dest)


def _classify(member_path: str, skills: tuple[str, ...]) -> str | None:
  """Return the atomic skill a clip belongs to, or None if it matches none.

  Two ways in: the file name says what the motion is, or the path does. The second is what
  reaches the subsets that number their trials rather than naming them.
  """
  path = member_path.replace("\\", "/")
  name = Path(path).name.lower()
  for skill in skills:
    if any(kw in name for kw in SKILL_KEYWORDS[skill]):
      return skill
    if any(part in path for part in SKILL_PATHS.get(skill, ())):
      return skill
  return None


def _curate(tarball: Path, output_dir: Path, skills: tuple[str, ...]) -> dict[str, int]:
  """Extract atomic ``.npz`` clips from a subset tarball into ``<skill>/`` folders."""
  counts = {skill: 0 for skill in skills}
  with tarfile.open(tarball, "r:bz2") as tar:
    for member in tar.getmembers():
      if not member.isfile() or not member.name.endswith(".npz"):
        continue
      skill = _classify(member.name, skills)
      if skill is None:
        continue
      src = tar.extractfile(member)
      if src is None:
        continue
      # Flatten and prefix with the subset + original folder so names stay
      # unique and traceable (e.g. ACCAD__Male2General__C3_-_Run_poses.npz).
      tag = re.sub(r"[^A-Za-z0-9]+", "_", tarball.stem.replace(".tar", "")).strip("_")
      flat = re.sub(r"[^A-Za-z0-9.]+", "_", member.name).strip("_")
      dest = output_dir / skill / f"{tag}__{flat}"
      dest.parent.mkdir(parents=True, exist_ok=True)
      with open(dest, "wb") as f:
        shutil.copyfileobj(src, f)
      counts[skill] += 1
  return counts


def main(
  output_dir: Path = Path("data/amass_atomic"),
  subsets: tuple[str, ...] = DEFAULT_SUBSETS,
  skills: tuple[str, ...] = tuple(SKILL_KEYWORDS),
  username: str | None = None,
  password: str | None = None,
  keep_tarballs: bool = False,
) -> None:
  """Download AMASS subsets and curate atomic SMPL-X clips by skill.

  Args:
    output_dir: Destination for the curated ``<skill>/*.npz`` clips (subset
      tarballs are cached under ``output_dir/tarballs``).
    subsets: AMASS subset names to pull (default ACCAD + KIT).
    skills: Atomic skills to keep. Defaults to all of walk/run/sprint/jump/
      crouch; see ``SKILL_KEYWORDS`` for the matched keywords.
    username: AMASS account email. Falls back to ``AMASS_USER``.
    password: AMASS account password. Falls back to ``AMASS_PASSWORD``.
    keep_tarballs: Keep the downloaded subset tarballs (otherwise removed once
      curated).
  """
  unknown = [s for s in skills if s not in SKILL_KEYWORDS]
  if unknown:
    raise SystemExit(
      f"Unknown skill(s): {', '.join(unknown)}. Available: {', '.join(SKILL_KEYWORDS)}"
    )

  username = username or os.environ.get("AMASS_USER")
  password = password or os.environ.get("AMASS_PASSWORD")
  if not username or not password:
    raise SystemExit(
      "AMASS credentials required. Register at https://amass.is.tue.mpg.de, "
      "then pass --username/--password or set AMASS_USER / AMASS_PASSWORD."
    )

  opener = _login(username, password)
  tar_dir = output_dir / "tarballs"

  totals = {skill: 0 for skill in skills}
  for subset in subsets:
    print(f"\n=== {subset} ===")
    tarball = tar_dir / f"{subset}.tar.bz2"
    _download_subset(opener, subset, tarball)
    counts = _curate(tarball, output_dir, skills)
    for skill, n in counts.items():
      totals[skill] += n
    print("  " + ", ".join(f"{skill}: {n}" for skill, n in counts.items()))
    if not keep_tarballs:
      tarball.unlink(missing_ok=True)

  summary = ", ".join(f"{skill}={n}" for skill, n in totals.items())
  print(
    f"\nDone. Curated atomic clips in {output_dir}/ ({summary}).\n"
    "These are robot-agnostic SMPL-X .npz files -- retarget them to your robot "
    "next. Cite AMASS (Mahmood et al., ICCV 2019) and the source subset(s)."
  )


if __name__ == "__main__":
  tyro.cli(main)
