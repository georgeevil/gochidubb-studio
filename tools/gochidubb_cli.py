#!/usr/bin/env python
"""GoChiDUBB Studio — CLI for local AI video dubbing.

Talks to a running GoChiDUBB server via HTTP. Server must already be up
(start.bat or `python server.py`). Default URL = http://localhost:8910,
override with env var `GOCHIDUBB_URL`.

Examples:
    # Dub a YouTube short into French
    python tools/gochidubb_cli.py dub https://youtu.be/abc --lang fr --wait

    # Re-dub job 5038e404 into 5 languages, stitched as showcase
    python tools/gochidubb_cli.py redub 5038e404 --langs es,fr,de,ja,pt --mode showcase --wait

    # Quick-test (separate side-by-side dubs)
    python tools/gochidubb_cli.py compare ./clip.mp4 --langs es,fr --trim 60

    # Show what's available
    python tools/gochidubb_cli.py system
    python tools/gochidubb_cli.py jobs --status complete
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime

# Allow running from anywhere by inserting our own dir into sys.path
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# Job labels and dub output may contain non-Latin chars (Russian, CJK, …).
# Russian Windows consoles default to cp1251 which can't render them, so
# reconfigure stdout to UTF-8 with a replacement fallback. Harmless on
# UTF-8 systems.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from gochidubb_client import GoChiDUBBClient, GoChiDUBBError, DEFAULT_URL


# ─────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────
def parse_when(value: str) -> float:
    """Parse a --at value into unix epoch seconds.

    Accepts either a numeric epoch ("1754300000") or an ISO-8601 datetime
    ("2026-08-05T02:00", "2026-08-05 02:00:00+03:00"). Naive datetimes are
    interpreted as LOCAL time — "schedule for 2 AM" means 2 AM on the
    machine running the server. Raises ValueError on anything else.
    """
    s = (value or "").strip()
    if not s:
        raise ValueError("empty schedule time")
    try:
        return float(s)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(
            f"Cannot parse {value!r} — use unix epoch seconds or ISO-8601 "
            f"(e.g. 2026-08-05T02:00)") from None
    # .timestamp() treats naive datetimes as local time, aware ones exactly.
    return dt.timestamp()


def _fmt_count(n) -> str:
    """1234567 -> '1.2M' (views column)."""
    if n is None:
        return "?"
    n = float(n)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return str(int(n))


def _fmt_dur(sec) -> str:
    """125 -> '2:05'; 3725 -> '1:02:05'."""
    if sec is None:
        return "?"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────
def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _print_job(j: dict, *, short: bool = False) -> None:
    if short:
        label = j.get("title") or j.get("source_label", "")
        line = f"{j.get('id', '?')[:8]}  {j.get('status', '?'):>12}  -> {j.get('target_lang', '?'):>3}  {label[:60]}"
        if j.get("error"):
            line += f"  err: {j['error'][:80]}"
        print(line)
    else:
        _print_json(j)


# ─────────────────────────────────────────────────────────────────────
# Subcommand handlers — each is async, takes args namespace
# ─────────────────────────────────────────────────────────────────────
async def cmd_dub(c: GoChiDUBBClient, a) -> None:
    scheduled_at = None
    if a.at:
        scheduled_at = parse_when(a.at)  # ValueError -> handled in main
    res = await c.submit_dub(
        a.source, a.lang,
        source_lang=a.source_lang, model=a.model,
        whisper_model=a.whisper, voice_preset=a.voice,
        voice_style=a.voice_style or "",
        tts_speed=a.tts_speed, speaker_mode=a.speakers,
        keep_bg=a.keep_bg, auto_denoise=a.auto_denoise,
        context_hint=a.context or "", wizard_mode=a.wizard_mode,
        mode=a.mode, scheduled_at=scheduled_at,
        voxcpm_cfg=a.voxcpm_cfg, voxcpm_steps=a.voxcpm_steps,
    )
    job_id = res.get("job_id")
    _print_json(res)
    if res.get("scheduled_at"):
        when = datetime.fromtimestamp(res["scheduled_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"[scheduled] job {job_id} will start at {when} (local)",
              file=sys.stderr)
        return  # --wait would idle until the scheduled time; poll later instead
    if a.wait and job_id:
        print(f"[wait] polling job {job_id}…", file=sys.stderr)
        final = await c.wait_for_job(job_id, timeout=a.wait_timeout)
        if final.get("status") == "complete":
            print(f"[done] {c.output_url(job_id)}")
        else:
            print(f"[fail] status={final.get('status')} err={final.get('error', '?')}",
                  file=sys.stderr)
            sys.exit(2)


async def cmd_compare(c: GoChiDUBBClient, a) -> None:
    res = await c.submit_compare(
        a.source, a.langs, trim_seconds=a.trim,
        source_lang=a.source_lang, model=a.model,
        whisper_model=a.whisper, voice_preset=a.voice,
        voice_style=a.voice_style or "",
        tts_speed=a.tts_speed, keep_bg=a.keep_bg,
        auto_denoise=a.auto_denoise,
        voxcpm_cfg=a.voxcpm_cfg, voxcpm_steps=a.voxcpm_steps,
    )
    _print_json(res)
    if a.wait and res.get("batch_id"):
        bid = res["batch_id"]
        print(f"[wait] polling batch {bid}…", file=sys.stderr)
        jobs = await c.wait_for_batch(bid, timeout=a.wait_timeout)
        for j in jobs:
            _print_job(j, short=True)


async def cmd_showcase(c: GoChiDUBBClient, a) -> None:
    res = await c.submit_showcase(
        a.source, a.langs, trim_seconds=a.trim,
        source_lang=a.source_lang, model=a.model,
        whisper_model=a.whisper, voice_preset=a.voice,
        voice_style=a.voice_style or "",
        tts_speed=a.tts_speed, keep_bg=a.keep_bg,
        auto_denoise=a.auto_denoise,
        voxcpm_cfg=a.voxcpm_cfg, voxcpm_steps=a.voxcpm_steps,
    )
    _print_json(res)
    if a.wait and res.get("batch_id"):
        bid = res["batch_id"]
        print(f"[wait] dubbing {len(res.get('target_langs', []))} langs…", file=sys.stderr)
        await c.wait_for_batch(bid, timeout=a.wait_timeout)
        print("[wait] stitching showcase reel…", file=sys.stderr)
        info = await c.wait_for_showcase(bid, timeout=a.wait_timeout)
        print(f"[done] {c.showcase_url(bid)}")
        if a.json:
            _print_json(info)


async def cmd_redub(c: GoChiDUBBClient, a) -> None:
    res = await c.redub(a.job_id, a.langs, mode=a.mode,
                        voxcpm_cfg=a.voxcpm_cfg, voxcpm_steps=a.voxcpm_steps)
    _print_json(res)
    if a.wait:
        if res.get("batch_id"):
            bid = res["batch_id"]
            await c.wait_for_batch(bid, timeout=a.wait_timeout)
            if a.mode == "showcase":
                await c.wait_for_showcase(bid, timeout=a.wait_timeout)
                print(f"[done] {c.showcase_url(bid)}")
            else:
                jobs = await c.list_jobs(batch_id=bid, limit=10)
                for j in jobs:
                    _print_job(j, short=True)
        elif res.get("job_id"):
            final = await c.wait_for_job(res["job_id"], timeout=a.wait_timeout)
            if final.get("status") == "complete":
                print(f"[done] {c.output_url(res['job_id'])}")
            else:
                print(f"[fail] {final.get('error', '?')}", file=sys.stderr)
                sys.exit(2)


async def cmd_status(c: GoChiDUBBClient, a) -> None:
    j = await c.get_job(a.job_id)
    if a.json:
        _print_json(j)
    else:
        _print_job(j)
        if j.get("status") == "complete":
            print(f"      url: {c.output_url(a.job_id)}")
        pub = j.get("publish")
        if isinstance(pub, dict):
            line = f"  publish: {pub.get('platform', '?')}/{pub.get('status', '?')}"
            if pub.get("url"):
                line += f" -> {pub['url']}"
            print(line)
        hint = j.get("download_hint")
        if isinstance(hint, dict) and hint:
            print(f"   rescue: {hint.get('summary', '')}")
            for cmd in hint.get("commands") or []:
                print(f"           {cmd.get('label', '?')}: {cmd.get('command', '')}")


async def cmd_jobs(c: GoChiDUBBClient, a) -> None:
    since = parse_when(a.since) if a.since else 0
    jobs = await c.list_jobs(limit=a.limit, status=a.status, batch_id=a.batch,
                             since=since)
    if a.json:
        _print_json(jobs)
        return
    if not jobs:
        print("(no jobs)")
        return
    for j in jobs:
        _print_job(j, short=True)


async def cmd_wait(c: GoChiDUBBClient, a) -> None:
    final = await c.wait_for_job(a.job_id, timeout=a.timeout)
    if a.json:
        _print_json(final)
    else:
        _print_job(final)
    if final.get("status") != "complete":
        sys.exit(2)


async def cmd_showcase_status(c: GoChiDUBBClient, a) -> None:
    info = await c.get_showcase(a.batch_id)
    _print_json(info)


async def cmd_showcase_rebuild(c: GoChiDUBBClient, a) -> None:
    res = await c.rebuild_showcase(a.batch_id)
    _print_json(res)
    if a.wait:
        await c.wait_for_showcase(a.batch_id, timeout=a.wait_timeout)
        print(f"[done] {c.showcase_url(a.batch_id)}")


async def cmd_cancel(c: GoChiDUBBClient, a) -> None:
    _print_json(await c.cancel_job(a.job_id))


async def cmd_rescue(c: GoChiDUBBClient, a) -> None:
    _print_json(await c.attach_source(a.job_id, a.video_file))


async def cmd_delete(c: GoChiDUBBClient, a) -> None:
    _print_json(await c.delete_job(a.job_id))


async def cmd_system(c: GoChiDUBBClient, a) -> None:
    s = await c.system_status()
    if a.json:
        _print_json(s)
        return
    # Pretty-printed health summary. Use ASCII so it works in cp1251/cp866
    # consoles (Russian Windows etc.) without UnicodeEncodeError.
    def ok(d):
        return "[ok]" if (isinstance(d, dict) and d.get("ok")) else "[--]"
    print(f"{ok(s.get('python'))} python   {s.get('python', {}).get('version', '?')}")
    print(f"{ok(s.get('ffmpeg'))} ffmpeg")
    print(f"{ok(s.get('gpu'))} gpu      {s.get('gpu', {}).get('name', '?')} "
          f"({s.get('gpu', {}).get('vram_gb', 0)} GB)")
    print(f"{ok(s.get('whisper'))} whisper  {s.get('whisper', {}).get('type', '?')}")
    print(f"{ok(s.get('voxcpm'))} voxcpm")
    print(f"{ok(s.get('demucs'))} demucs")
    ollama = s.get("ollama") or s.get("ollama_binary") or {}
    print(f"{ok(ollama)} ollama")
    models = ollama.get("models", []) if isinstance(ollama, dict) else []
    if models:
        names = [m if isinstance(m, str) else m.get("name", "") for m in models]
        print(f"          models: {', '.join(names[:6])}{'...' if len(names) > 6 else ''}")


async def cmd_languages(c: GoChiDUBBClient, a) -> None:
    print(",".join(await c.list_languages()))


async def cmd_models(c: GoChiDUBBClient, a) -> None:
    for m in await c.list_models():
        print(m)


async def cmd_voices(c: GoChiDUBBClient, a) -> None:
    for v in await c.list_voices():
        if isinstance(v, dict):
            print(f"{v.get('name', '?'):<20} {v.get('lang', '?'):<6} {v.get('description', '')}")
        else:
            print(v)


def _parse_cast(pairs) -> dict:
    """--set SPEAKER_00=male_deep, repeated. Kept out of the client so the
    server stays the single validator of voice ids."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects SPEAKER=voice, got {item!r}")
        sp, voice = item.split("=", 1)
        out[sp.strip()] = voice.strip()
    return out


async def cmd_cast(c: GoChiDUBBClient, a) -> None:
    """Show, set, or audition the voice assigned to each speaker."""
    cast = _parse_cast(a.set)
    if cast:
        res = await c.set_voice_casting(a.job_id, cast)
        if a.json:
            _print_json(res)
        else:
            print(f"cast saved ({res.get('checkpoints_updated', 0)} checkpoints)")

    if a.preview:
        res = await c.preview_voice_casting(
            a.job_id, cast or None, per_speaker=a.per_speaker)
        if a.json:
            _print_json(res)
        else:
            print(f"previewed {len(res.get('samples', []))} line(s) "
                  f"in {res.get('took_sec', '?')}s")
            for smp in res.get("samples", []):
                print(f"  {smp['speaker']:<12} {smp['voice']:<28} {smp['url']}")
        return

    if cast:
        return

    data = await c.get_voice_casting(a.job_id)
    if a.json:
        _print_json(data)
        return
    current = data.get("map") or {}
    print(f"cast — job {a.job_id}  (target {data.get('target_lang', '?')}, "
          f"{data.get('status', '?')})")
    for sp in data.get("speakers", []):
        print(f"  {sp['speaker']:<12} {sp['segments']:>5} lines "
              f"{round(sp['share'] * 100):>3}%  "
              f"{'ref' if sp['has_source_ref'] else 'NO REF':<6}  "
              f"→ {current.get(sp['speaker'], 'source')}")
    print("\navailable voices:")
    for v in data.get("voices", []):
        print(f"  {v['id']:<22} {v['kind']:<8} {v['name']}")
    print("  source                 —        keep the voice cloned from the video")
    print("  design:<description>   design   describe one in words")


async def cmd_continue(c: GoChiDUBBClient, a) -> None:
    """Resume a job parked at a review gate."""
    res = await c.continue_job(a.job_id)
    if a.json:
        _print_json(res)
    else:
        print(f"resuming {a.job_id} from {res.get('resuming_from', '?')}")


async def cmd_quality(c: GoChiDUBBClient, a) -> None:
    q = await c.get_quality(a.job_id)
    if a.json:
        _print_json(q)
        return
    roll = q.get("rollup") or {}
    overall = roll.get("overall")
    print(f"quality — job {a.job_id}  "
          f"({q.get('n_segments', '?')} segments from '{q.get('segments_from', '?')}')")
    print(f"  overall     {overall if overall is not None else 'n/a':>4}"
          f"{'/100' if overall is not None else ''}")
    for name, st in (roll.get("stages") or {}).items():
        score = st.get("score") if isinstance(st, dict) else None
        shown = f"{score:>4}/100" if score is not None else " n/a (not measured)"
        print(f"  {name:<12}{shown}")
    verdicts = roll.get("verdicts") or []
    if verdicts:
        print("verdicts:")
        for v in verdicts:
            segs = v.get("segment_indices") or []
            seg_txt = f"  segments={','.join(str(s) for s in segs[:12])}" if segs else ""
            act = f"  -> {v['suggested_action']}" if v.get("suggested_action") else ""
            print(f"  [{v.get('severity', '?'):<4}] {v.get('stage', '?')}: "
                  f"{v.get('message', '')}{seg_txt}{act}")
    else:
        print("verdicts: none — nothing actionable")


async def cmd_audit(c: GoChiDUBBClient, a) -> None:
    rep = await c.audit(a.job_id)
    if a.json:
        _print_json(rep)
        return
    counts = rep.get("counts") or {}
    print(f"audit — job {a.job_id}  ok={rep.get('ok')}  "
          f"loss={counts.get('loss', 0)} warn={counts.get('warn', 0)} "
          f"info={counts.get('info', 0)}")
    for row in rep.get("stages") or []:
        print(f"  {row.get('stage', '?'):<14} segments={row.get('segments', '?'):<4} "
              f"words={row.get('words', '?')}")
    for f in rep.get("findings") or []:
        print(f"  [{f.get('severity', '?'):<4}] {f.get('stage', '?')}: {f.get('summary', '')}")
        if f.get("detail"):
            print(f"         {f['detail']}")


def _print_publish(pub: dict) -> None:
    print(f"  platform:  {pub.get('platform', '?')}")
    print(f"  status:    {pub.get('status', '?')}")
    print(f"  title:     {pub.get('title', '')}")
    desc = (pub.get("description") or "").strip()
    if desc:
        first = desc.splitlines()[0]
        print(f"  descr:     {first[:100]}{'…' if len(desc) > 100 else ''}")
    print(f"  file:      {pub.get('file', '?')}")
    if pub.get("url"):
        print(f"  url:       {pub['url']}")
    if pub.get("error"):
        print(f"  error:     {pub['error']}")
    verdict = pub.get("duplicate_verdict")
    if verdict:
        print(f"  dup check: {verdict}")
    for m in (pub.get("duplicate_warnings") or [])[:5]:
        print(f"    ~ score={m.get('score', '?')} dur={_fmt_dur(m.get('duration'))} "
              f"{(m.get('title') or '')[:70]}  {m.get('url', '')}")


async def cmd_publish(c: GoChiDUBBClient, a) -> None:
    res = await c.publish_stage(a.job_id, platform=a.platform,
                                export_preset=a.preset)
    if a.json:
        _print_json(res)
        return
    if res.get("warning"):
        print(f"[warn] {res['warning']}", file=sys.stderr)
    pub = res.get("publish") or {}
    print(f"staged job {a.job_id} for publishing — review then run:")
    print(f"  gochidubb approve {a.job_id}  (this triggers the actual upload)")
    _print_publish(pub)


async def cmd_approve(c: GoChiDUBBClient, a) -> None:
    res = await c.publish_approve(a.job_id, title=a.title,
                                  description=a.description)
    if a.json:
        _print_json(res)
        return
    pub = res.get("publish") or {}
    print(f"approved — upload queued (status={pub.get('status', '?')})")
    _print_publish(pub)


async def cmd_publish_cancel(c: GoChiDUBBClient, a) -> None:
    _print_json(await c.publish_cancel(a.job_id))


async def cmd_publish_inbox(c: GoChiDUBBClient, a) -> None:
    pending = await c.publish_pending()
    if a.json:
        _print_json(pending)
        return
    if not pending:
        print("(publish inbox empty)")
        return
    for p in pending:
        pub = p.get("publish") or {}
        print(f"{p.get('job_id', '?')[:8]}  {pub.get('platform', '?'):<4} "
              f"{pub.get('status', '?'):>9}  dup={pub.get('duplicate_verdict', '?'):<16} "
              f"{(p.get('title') or '')[:60]}")


async def cmd_scout(c: GoChiDUBBClient, a) -> None:
    res = await c.scout_trending(category=a.category, country=a.country,
                                 limit=a.limit, include_shorts=a.shorts)
    if a.json:
        _print_json(res)
        return
    cands = res.get("candidates") or []
    if not cands:
        print(f"(no candidates — source: {res.get('source', '?')})")
        return
    print(f"trending via {res.get('source', '?')}:")
    for cand in cands:
        badges = []
        if cand.get("is_music"):
            badges.append("music")
        if cand.get("is_short"):
            badges.append("short")
        if cand.get("already_processed"):
            badges.append(f"processed:{(cand.get('existing_job_id') or '?')[:8]}")
        badge_txt = f" [{','.join(badges)}]" if badges else ""
        print(f"{cand.get('rank', '?'):>3}. {cand.get('video_id', '?'):<12} "
              f"{_fmt_count(cand.get('view_count')):>7} views  "
              f"{_fmt_dur(cand.get('duration_sec')):>7}  "
              f"{(cand.get('title') or '')[:52]:<52}  "
              f"{(cand.get('channel') or '')[:24]}{badge_txt}")


async def cmd_scout_dub(c: GoChiDUBBClient, a) -> None:
    scheduled_at = parse_when(a.at) if a.at else None
    res = await c.scout_dub(a.video, a.lang, mode=a.mode,
                            scheduled_at=scheduled_at)
    _print_json(res)


async def cmd_check_dup(c: GoChiDUBBClient, a) -> None:
    res = await c.check_duplicate(a.title, duration_sec=a.duration,
                                  alt_title=a.alt_title)
    if a.json:
        _print_json(res)
        return
    print(f"verdict: {res.get('verdict', '?')}")
    for m in res.get("matches") or []:
        print(f"  score={m.get('score', '?')} dur={_fmt_dur(m.get('duration'))} "
              f"{(m.get('title') or '')[:70]}  {m.get('url', '')}")


# ─────────────────────────────────────────────────────────────────────
# argparse setup
# ─────────────────────────────────────────────────────────────────────
def _add_common_dub_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source-lang", default="auto", help="Source language code or 'auto'")
    p.add_argument("--model", help="Ollama translation model (e.g. aya-expanse:8b)")
    p.add_argument("--whisper", default="large-v3", help="Whisper model size")
    p.add_argument("--voice", default="auto", help="Voice preset name")
    p.add_argument("--voice-style", default="",
                   help="Free-text voice style hint (overrides preset cloning)")
    p.add_argument("--tts-speed", default="balanced",
                   choices=["fast", "balanced", "quality"])
    p.add_argument("--keep-bg", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep background music/SFX (default: on; --no-keep-bg to disable)")
    p.add_argument("--auto-denoise", action="store_true",
                   help="Denoise the voice reference before cloning")
    p.add_argument("--voxcpm-cfg", type=float, default=0.0,
                   help="Per-job VoxCPM guidance, 1.0-3.0 "
                        "(default 0 = use the server's global setting)")
    p.add_argument("--voxcpm-steps", type=int, default=0,
                   help="Per-job VoxCPM inference steps, 4-24 "
                        "(default 0 = use the server's global setting)")
    p.add_argument("--wait", action="store_true", help="Block until job(s) finish")
    p.add_argument("--wait-timeout", type=float, default=1800.0,
                   help="Seconds before --wait gives up (default 1800)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gochidubb",
        description="Command-line interface for GoChiDUBB Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Server URL: env GOCHIDUBB_URL (default %s)" % DEFAULT_URL,
    )
    p.add_argument("--url", default=DEFAULT_URL, help="Override server URL for this call")
    sub = p.add_subparsers(dest="cmd", required=True)

    # dub
    s = sub.add_parser("dub", help="Submit a single-language dub")
    s.add_argument("source", help="Video URL or local file path")
    s.add_argument("--lang", required=True, help="Target language code (e.g. fr, es, de)")
    s.add_argument("--context", help="Optional translator hint (genre, tone)")
    s.add_argument("--speakers", default="main", choices=["main", "all", "single"])
    s.add_argument("--mode", default="dub", choices=["dub", "reupload"],
                   help="reupload = download + remux only (music videos)")
    s.add_argument("--wizard-mode", default="auto",
                   choices=["auto", "review_translation", "review_transcript",
                            "review_voices"],
                   help="Pause for human review at a checkpoint")
    s.add_argument("--at", metavar="WHEN",
                   help="Schedule start: unix epoch or ISO datetime "
                        "(naive = local time), e.g. 2026-08-05T02:00")
    _add_common_dub_opts(s)
    s.set_defaults(handler=cmd_dub)

    # compare (quick test)
    s = sub.add_parser("compare", help="N separate dubs side-by-side (Quick Test)")
    s.add_argument("source")
    s.add_argument("--langs", required=True, help="Comma-separated 2-6 codes")
    s.add_argument("--trim", type=int, default=60, help="Trim seconds (15-120)")
    _add_common_dub_opts(s)
    s.set_defaults(handler=cmd_compare)

    # showcase
    s = sub.add_parser("showcase", help="Multilingual stitched reel (one video, N segments)")
    s.add_argument("source")
    s.add_argument("--langs", required=True, help="Comma-separated 2-6 codes")
    s.add_argument("--trim", type=int, default=60)
    s.add_argument("--json", action="store_true")
    _add_common_dub_opts(s)
    s.set_defaults(handler=cmd_showcase)

    # redub
    s = sub.add_parser("redub", help="Re-dub an existing job's source in new languages")
    s.add_argument("job_id", help="Original job ID")
    s.add_argument("--langs", required=True, help="Comma-separated target languages")
    s.add_argument("--mode", default="compare", choices=["single", "compare", "showcase"])
    s.add_argument("--voxcpm-cfg", type=float,
                   help="Override VoxCPM guidance for the new dubs "
                        "(omit = inherit from the original job, 0 = global)")
    s.add_argument("--voxcpm-steps", type=int,
                   help="Override VoxCPM inference steps for the new dubs "
                        "(omit = inherit from the original job, 0 = global)")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--wait-timeout", type=float, default=1800.0)
    s.set_defaults(handler=cmd_redub)

    # status / jobs / wait
    s = sub.add_parser("status", help="Show one job's status")
    s.add_argument("job_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_status)

    s = sub.add_parser("jobs", help="List jobs (default: 50 newest)")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--status", help="Status filter, comma-separated OK (queued,running)")
    s.add_argument("--batch", help="Filter by batch_id")
    s.add_argument("--since", help="Only jobs created at/after this time "
                                   "(unix epoch or ISO datetime)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_jobs)

    s = sub.add_parser("wait", help="Block until a job finishes")
    s.add_argument("job_id")
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_wait)

    # showcase utils
    s = sub.add_parser("showcase-status", help="Showcase assembly status by batch_id")
    s.add_argument("batch_id")
    s.set_defaults(handler=cmd_showcase_status)

    s = sub.add_parser("showcase-rebuild", help="Re-run showcase stitch step")
    s.add_argument("batch_id")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--wait-timeout", type=float, default=600.0)
    s.set_defaults(handler=cmd_showcase_rebuild)

    # quality / audit
    s = sub.add_parser("cast", help="Assign one voice per speaker, and audition it")
    s.add_argument("job_id")
    s.add_argument("--set", action="append", metavar="SPEAKER=VOICE",
                   help="Assign a voice. VOICE is a preset id, file:<name>, "
                        "design:<description>, or source. Repeatable.")
    s.add_argument("--preview", action="store_true",
                   help="Synthesize a real line per speaker and print the URLs")
    s.add_argument("--per-speaker", type=int, default=1, dest="per_speaker",
                   help="Lines to preview per speaker (1-3, default 1)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_cast)

    s = sub.add_parser("continue", help="Resume a job parked at a review gate")
    s.add_argument("job_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_continue)

    s = sub.add_parser("quality", help="Per-stage quality scores + actionable verdicts")
    s.add_argument("job_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_quality)

    s = sub.add_parser("audit", help="Artifact-loss audit (word coverage, idx integrity)")
    s.add_argument("job_id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_audit)

    # publish
    s = sub.add_parser("publish", help="Stage a finished job for publishing (no upload)")
    s.add_argument("job_id")
    s.add_argument("--platform", default="vk", help="Target platform (default vk)")
    s.add_argument("--preset", help="Export preset to render first (e.g. vk_1080p)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_publish)

    s = sub.add_parser("approve", help="Approve a staged publish — TRIGGERS the upload")
    s.add_argument("job_id")
    s.add_argument("--title", help="Override the staged title")
    s.add_argument("--description", help="Override the staged description")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_approve)

    s = sub.add_parser("publish-cancel", help="Withdraw a staged/approved publish")
    s.add_argument("job_id")
    s.set_defaults(handler=cmd_publish_cancel)

    s = sub.add_parser("publish-inbox", help="Review inbox: publishes awaiting action")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_publish_inbox)

    # scout
    s = sub.add_parser("scout", help="Trending YouTube candidates (mostviewed.today)")
    s.add_argument("--country", help="Country code (e.g. us, ru, de)")
    s.add_argument("--category", help="Category slug (e.g. music, gaming) or 'shorts'")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--shorts", action="store_true", help="Include YouTube Shorts")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_scout)

    s = sub.add_parser("scout-dub", help="Send a scout candidate into the dub pipeline")
    s.add_argument("video", help="YouTube video_id or URL")
    s.add_argument("lang", help="Target language code")
    s.add_argument("--mode", default="auto", choices=["auto", "dub", "reupload"],
                   help="auto = reupload for music candidates, else dub")
    s.add_argument("--at", metavar="WHEN",
                   help="Schedule start (unix epoch or ISO datetime)")
    s.set_defaults(handler=cmd_scout_dub)

    s = sub.add_parser("check-dup", help="Does a similar video already exist on VK?")
    s.add_argument("title")
    s.add_argument("--duration", type=float, help="Video duration in seconds")
    s.add_argument("--alt-title", help="Alternative (e.g. original-language) title")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_check_dup)

    # admin
    s = sub.add_parser("cancel", help="Cancel a running job")
    s.add_argument("job_id")
    s.set_defaults(handler=cmd_cancel)
    s = sub.add_parser("rescue", help="Attach a manually-downloaded video "
                                      "to a failed job and resume it")
    s.add_argument("job_id")
    s.add_argument("video_file")
    s.set_defaults(handler=cmd_rescue)
    s = sub.add_parser("delete", help="Delete a job and its files")
    s.add_argument("job_id")
    s.set_defaults(handler=cmd_delete)
    s = sub.add_parser("system", help="System health check")
    s.add_argument("--json", action="store_true")
    s.set_defaults(handler=cmd_system)
    s = sub.add_parser("languages", help="List supported language codes")
    s.set_defaults(handler=cmd_languages)
    s = sub.add_parser("models", help="List installed Ollama models")
    s.set_defaults(handler=cmd_models)
    s = sub.add_parser("voices", help="List voice presets")
    s.set_defaults(handler=cmd_voices)
    return p


async def _main_async(args) -> int:
    async with GoChiDUBBClient(base_url=args.url) as c:
        try:
            await args.handler(c, args)
            return 0
        except (GoChiDUBBError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
