from __future__ import annotations

"""
Cookie policy and anti-bot request shaping.

Three jobs:

1) WHEN TO SEND THE COOKIE — a public reel, a public tweet or a public
   YouTube video downloads perfectly well logged out. Sending the session on
   those requests buys nothing and costs a lot: every authenticated request
   ties this server's IP to one account, and that is exactly the pattern the
   platforms score as "bot". So cookies are used when the content actually
   needs them (stories, private posts, age gates) or as a FALLBACK after the
   anonymous attempt is refused, not as the default first move.

2) HOW THE REQUEST LOOKS — a real browser is recognised by its TLS handshake
   first and its headers second. With curl_cffi present yt-dlp can copy
   Chrome's handshake; the header set here matches the same browser so the
   two halves tell the same story. The user agent is stable per platform per
   day rather than random per request: a client whose fingerprint changes
   mid-session is more suspicious than a boring one.

3) BACKING OFF — when a platform does answer with a rate limit or a bot
   check, the account is what gets burned. The cooldown below drops that
   platform back to anonymous requests for a while and slows them down,
   instead of retrying with the session until it is locked.
"""

import hashlib
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from .storage import read_json, write_json_atomic
from .utils import instagram_story_kind, platform_name

# Cookies buy nothing here — the content is public and the session only adds
# a fingerprint. Cookies stay available as a fallback attempt.
_PUBLIC_FIRST = {
    "YouTube", "YouTube Music", "TikTok", "X/Twitter", "Twitter",
    "Reddit", "Pinterest", "Spotify",
}

# Almost nothing on these is readable logged out, so the anonymous attempt is
# just a wasted round trip.
_COOKIE_FIRST = {"Facebook"}

# URL shapes that genuinely need a session, whatever the platform default is.
_LOGIN_WALLED_PATTERNS = (
    r"/stories/",
    r"/s/aGlnaGxpZ2h0",      # highlight permalinks
    r"/private/",
)


def _platform(url: str) -> str:
    return platform_name(url)


def needs_cookies(url: str) -> bool:
    """True when the link cannot be fetched without a session at all."""
    path = (urlparse(url).path or "")
    if instagram_story_kind(url):
        return True
    return any(re.search(pattern, path, re.IGNORECASE) for pattern in _LOGIN_WALLED_PATTERNS)


@contextmanager
def private_cookies(cookies_file: Path | None, tag: str) -> Iterator[Path | None]:
    """
    Hands out a throwaway copy of the cookie file for one yt-dlp run.

    yt-dlp writes the jar back when it closes, and it does that by opening
    the file in "w" — truncate first, write after. Three download workers and
    the link preview all pointed at the SAME data/cookies.txt, so one job
    truncating while another read meant an empty or half-written session out
    of nowhere: every platform "logged out" at once, for no reason anybody
    could see in the logs.

    Working from a copy means nothing writes the shared file any more. The
    cost is that cookies refreshed during a download are dropped; the file is
    maintained by the admin anyway, and a stale cookie is a far smaller
    problem than a shredded one.
    """
    if not cookies_file:
        yield None
        return

    source = Path(cookies_file)
    if not source.exists():
        yield None
        return

    handle, temp_name = tempfile.mkstemp(prefix=f"cookies-{tag}-", suffix=".txt")
    os.close(handle)
    copy = Path(temp_name)
    try:
        shutil.copy2(source, copy)
        os.chmod(copy, 0o600)
        yield copy
    finally:
        copy.unlink(missing_ok=True)


class CookiePreference:
    """
    Remembers which request style actually worked per platform.

    The static policy above is a good first guess, not a fact: this server's
    IP can be anonymous-friendly on one platform and blocked on another, and
    that changes over weeks. So whichever style last produced a file goes
    first next time, and the guess is only used until there is evidence.
    """

    TTL = 12 * 3600

    # A platform that refuses anonymous requests is refusing this server's
    # IP, and that verdict outlives a single day. Re-testing it every 12
    # hours costs a wasted request and, worse, another bot-check strike
    # against the address, so a block is remembered for a week.
    BLOCK_TTL = 7 * 24 * 3600

    def __init__(self, data_dir: Path):
        self.file = Path(data_dir) / "cookie_pref.json"

    def _load(self) -> dict[str, Any]:
        data = read_json(self.file, {"platforms": {}})
        if not isinstance(data, dict) or not isinstance(data.get("platforms"), dict):
            return {"platforms": {}}
        return data

    def _entry(self, platform: str) -> dict[str, Any]:
        entry = self._load()["platforms"].get(platform)
        return entry if isinstance(entry, dict) else {}

    def _write(self, platform: str, changes: dict[str, Any]) -> None:
        if not platform:
            return
        try:
            data = self._load()
            entry = data["platforms"].get(platform)
            entry = dict(entry) if isinstance(entry, dict) else {}
            entry.update(changes)
            data["platforms"][platform] = entry
            write_json_atomic(self.file, data)
        except Exception:
            pass

    def record_success(self, platform: str, used_cookies: bool) -> None:
        changes: dict[str, Any] = {"cookies": bool(used_cookies), "at": time.time()}
        # Getting a file anonymously is proof the block is gone.
        if not used_cookies:
            changes["anon_blocked_at"] = 0
        self._write(platform, changes)

    def record_anonymous_block(self, platform: str) -> None:
        """The platform answered a logged-out request with a bot check."""
        self._write(platform, {"anon_blocked_at": time.time()})

    def anonymous_blocked(self, platform: str) -> bool:
        entry = self._entry(platform)
        blocked_at = float(entry.get("anon_blocked_at") or 0)
        return bool(blocked_at) and time.time() - blocked_at <= self.BLOCK_TTL

    def preferred(self, platform: str) -> bool | None:
        if self.anonymous_blocked(platform):
            return True
        entry = self._entry(platform)
        if not entry or "cookies" not in entry:
            return None
        if time.time() - float(entry.get("at", 0)) > self.TTL:
            return None
        return bool(entry.get("cookies"))


def cookie_order(url: str, *, data_dir: Path | None = None) -> tuple[bool, ...]:
    """
    The order the two request styles are tried in: True = with cookies.

    Both are always present — the fallback is what keeps a public-first
    policy from losing content — only the order changes.
    """
    if needs_cookies(url):
        return (True, False)

    platform = _platform(url)

    if data_dir is not None:
        learned = CookiePreference(data_dir).preferred(platform)
        if learned is not None:
            return (learned, not learned)

    if platform in _COOKIE_FIRST:
        return (True, False)
    if platform in _PUBLIC_FIRST:
        return (False, True)

    # Instagram and anything unknown: public content first, session second.
    return (False, True)


# ── Request shaping ──────────────────────────────────────────────────────────

# One family, several plausible machines. Chrome on Windows/macOS is the most
# common thing a platform sees, so it is the least interesting thing to be.
#
# The Chrome VERSION is pinned rather than rotated, and it is not the newest
# one on purpose. TikTok serves a different page per user agent version, and
# yt-dlp's extractor can only read one of them: measured here, 139 works
# while 138, 140 and 145 all come back as "Unexpected response from webpage
# request" or "Unable to download webpage". Only the operating system
# rotates, which every version handled identically.
#
# So when TikTok starts failing across the board after a yt-dlp upgrade,
# this is the first thing to re-check.
_CHROME_VERSION = "139"

_USER_AGENTS = (
    (f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     f"(KHTML, like Gecko) Chrome/{_CHROME_VERSION}.0.0.0 Safari/537.36", '"Windows"'),
    (f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
     f"(KHTML, like Gecko) Chrome/{_CHROME_VERSION}.0.0.0 Safari/537.36", '"macOS"'),
    (f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
     f"(KHTML, like Gecko) Chrome/{_CHROME_VERSION}.0.0.0 Safari/537.36", '"Linux"'),
)


def _agent_for(platform: str) -> tuple[str, str]:
    """Stable per platform per day: same browser all day, not a new one per request."""
    day = int(time.time() // 86400)
    digest = hashlib.sha256(f"{platform}:{day}".encode()).digest()
    return _USER_AGENTS[digest[0] % len(_USER_AGENTS)]


def browser_headers(url: str) -> dict[str, str]:
    """A header set that matches the impersonated browser."""
    agent, platform_hint = _agent_for(_platform(url))
    major = _CHROME_VERSION
    return {
        "User-Agent": agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": (
            f'"Chromium";v="{major}", "Not;A=Brand";v="24", '
            f'"Google Chrome";v="{major}"'
        ),
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": platform_hint,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


# Platforms where the TLS fingerprint is checked. YouTube is left on yt-dlp's
# own handler on purpose: its extractor does its own client negotiation and
# impersonation gains nothing there.
_IMPERSONATE_PLATFORMS = {
    "Instagram", "TikTok", "X/Twitter", "Twitter", "Facebook",
    "Reddit", "Pinterest",
}


# Whether this install can actually impersonate anything. Asking yt-dlp
# costs a handler probe, so it is asked once per process.
_IMPERSONATION_AVAILABLE: bool | None = None


def _impersonation_available(target: Any) -> bool:
    """
    Is the target really usable here?

    Handing yt-dlp a target it cannot serve does not degrade — it raises
    before the request is made, so an install without curl_cffi (or with a
    version yt-dlp silently refuses, see requirements.txt) would fail every
    social download instead of just losing the disguise.
    """
    global _IMPERSONATION_AVAILABLE

    if _IMPERSONATION_AVAILABLE is None:
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                available = [item[0] for item in ydl._get_available_impersonate_targets()]
            _IMPERSONATION_AVAILABLE = any(target in candidate for candidate in available)
        except Exception:
            _IMPERSONATION_AVAILABLE = False

    return _IMPERSONATION_AVAILABLE


def impersonate_target(url: str) -> Any | None:
    """yt-dlp ImpersonateTarget for this link, or None when not applicable."""
    if _platform(url) not in _IMPERSONATE_PLATFORMS:
        return None
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
    except Exception:
        return None

    target = ImpersonateTarget("chrome")
    return target if _impersonation_available(target) else None


def extractor_args(url: str) -> dict[str, Any]:
    """
    Extractor-specific arguments for this URL. Empty when none apply.

    YouTube's normal clients now hand out their https (DASH) streams only
    against a GVS PO token; without one they are dropped and all that is
    left is HLS. That is the expensive route: no audio-only stream exists
    there at all, so an audio job downloaded a 9.6 MB muxed video to keep a
    3 MB song, and every file arrived in 26 fragments — the shape that also
    trips yt-dlp's parallel-fragment bug.

    The embedded player still answers without a token (33 formats against
    11, audio-only among them). It is ADDED to the default clients rather
    than replacing them, so a video that cannot be embedded still gets an
    answer from the usual ones; the extra client costs about half a second.
    """
    host = (urlparse(url).netloc or "").lower()
    if "youtube.com" in host or host.endswith("youtu.be"):
        return {"youtube": {"player_client": ["web_embedded", "default"]}}
    return {}


def pacing(url: str) -> dict[str, Any]:
    """
    A short pause between requests on the platforms that count them.

    Costs about a second on a normal download and keeps a burst of metadata
    requests from looking like a scraper.
    """
    if _platform(url) in _IMPERSONATE_PLATFORMS:
        return {"sleep_interval_requests": 1}
    return {}


# ── Cooldown after a rate limit / bot check ──────────────────────────────────

_BOT_CHECK_PATTERNS = (
    r"rate.?limit", r"too many requests", r"http error 429",
    r"sign in to confirm", r"confirm you'?re not a bot",
    r"unusual activity", r"suspicious", r"temporarily blocked",
    r"please wait a few minutes", r"challenge_required", r"checkpoint_required",
)

COOLDOWN_SECONDS = 45 * 60


def is_bot_check_error(message: str) -> bool:
    """Did the platform answer with a rate limit or a bot check?"""
    lowered = str(message or "").lower()
    return any(re.search(pattern, lowered) for pattern in _BOT_CHECK_PATTERNS)


class CookieCooldown:
    """
    Remembers which platform just pushed back, so the next few jobs stay
    anonymous there instead of spending the session on a wall.

    File backed because downloads run in separate processes.
    """

    def __init__(self, data_dir: Path):
        self.file = Path(data_dir) / "cookie_cooldown.json"

    def _load(self) -> dict[str, Any]:
        data = read_json(self.file, {"platforms": {}})
        if not isinstance(data, dict) or not isinstance(data.get("platforms"), dict):
            return {"platforms": {}}
        return data

    def mark(self, platform: str, reason: str = "") -> None:
        if not platform:
            return
        try:
            data = self._load()
            data["platforms"][platform] = {
                "until": time.time() + COOLDOWN_SECONDS,
                "reason": str(reason)[:200],
            }
            write_json_atomic(self.file, data)
        except Exception:
            pass

    def active(self, platform: str) -> bool:
        entry = self._load()["platforms"].get(platform)
        if not isinstance(entry, dict):
            return False
        return float(entry.get("until", 0)) > time.time()

    def entries(self) -> dict[str, Any]:
        now = time.time()
        return {
            name: entry
            for name, entry in self._load()["platforms"].items()
            if isinstance(entry, dict) and float(entry.get("until", 0)) > now
        }
