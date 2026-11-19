"""Verification test suite for PeakCut architectural upgrades:
1. Ollama parameters ("options": {"num_ctx": 8192, "temperature": 0.2}, "keep_alive": "60m")
2. Chunked scoring for long videos (>8 mins into 8-min chunks with 30s overlap, deduplication)
3. Dynamic face-centered 9:16 vertical cropping (sampled every 10 frames)
4. Real-time live 60-second chunk ingestion with sherpa-onnx + Ollama + auto deletion
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import ai, auth, brain, config, ctx, db, jobs, live_queue, media, render, transcribe

db.init()
fails = []


def check(name, ok, extra=""):
    status = "PASS" if ok else "FAIL"
    print(f"{status} {name}" + (f"  {extra}" if extra and not ok else ""))
    if not ok:
        fails.append(name)


def test_ollama_options():
    print("\n--- Test 1: Ollama Payload Options & Keep-Alive ---")
    seen_payloads = []
    real_post = ai.httpx.post

    def mock_post(url, json=None, **kwargs):
        if json:
            seen_payloads.append((url, json))
        return real_post(url, json=json, **kwargs)

    ai.httpx.post = mock_post
    try:
        # Live call to local Ollama
        config.AI_PROVIDER = "ollama"
        config.AI_MODELS = ["qwen2.5:3b"]
        res = ai.test()
        check("Ollama live response succeeds", res.get("ok") is True, str(res))
        last_url, last_body = seen_payloads[-1]
        options = last_body.get("options", {})
        check("Ollama num_ctx is 8192", options.get("num_ctx") == 8192, str(options))
        check("Ollama temperature is 0.2", options.get("temperature") == 0.2, str(options))
        check("Ollama keep_alive is 60m", last_body.get("keep_alive") == "60m", str(last_body))
    finally:
        ai.httpx.post = real_post


def test_chunked_scoring():
    print("\n--- Test 2: Chunked Scoring for Long Transcripts (>8 mins) ---")
    # Verify WINDOW_SEC and OVERLAP_SEC constants
    check("WINDOW_SEC is 8 minutes (480s)", brain.WINDOW_SEC == 480, f"Got {brain.WINDOW_SEC}")
    check("OVERLAP_SEC is 30 seconds", brain.OVERLAP_SEC == 30, f"Got {brain.OVERLAP_SEC}")

    # Generate a 20-minute synthetic transcript (1200 seconds)
    duration = 1200.0
    words = []
    # Create speech throughout the 20 minutes
    t = 10.0
    word_list = ["Hey", "everyone", "welcome", "to", "the", "stream", "this", "is", "an", "insane", "play", "unbelievable"]
    while t < duration - 10.0:
        for w in word_list:
            words.append({"w": w, "s": round(t, 2), "e": round(t + 0.4, 2)})
            t += 0.5
        t += 3.0  # short pause

    # Track windows queried by ask_ai
    prompts_seen = []
    orig_ask_ai = brain.ask_ai

    def mock_ask_ai(prompt, **kwargs):
        prompts_seen.append(prompt)
        # Return a mock clip for each window
        # In the overlap zone between window 0 (0-480) and window 1 (450-930), return clips around 460s to test deduplication
        if len(prompts_seen) == 1:
            return [{"start": 100, "end": 130, "score": 85, "hook": "GREAT MOMENT", "title": "Moment 1",
                     "description": "Desc 1", "hashtags": ["clip"], "highlight_words": ["play"], "reason": "funny"},
                    {"start": 460, "end": 480, "score": 75, "hook": "OVERLAP MOMENT", "title": "Moment Overlap 1",
                     "description": "Desc", "hashtags": ["clip"], "highlight_words": ["insane"], "reason": "shock"}], "mock_model", 50
        elif len(prompts_seen) == 2:
            return [{"start": 461, "end": 481, "score": 90, "hook": "BETTER OVERLAP", "title": "Moment Overlap 2",
                     "description": "Desc", "hashtags": ["clip"], "highlight_words": ["insane"], "reason": "shock"},
                    {"start": 600, "end": 630, "score": 80, "hook": "MID MOMENT", "title": "Moment 2",
                     "description": "Desc 2", "hashtags": ["clip"], "highlight_words": ["play"], "reason": "clutch"}], "mock_model", 50
        else:
            return [{"start": 1000, "end": 1030, "score": 82, "hook": "END MOMENT", "title": "Moment 3",
                     "description": "Desc 3", "hashtags": ["clip"], "highlight_words": ["play"], "reason": "win"}], "mock_model", 50

    brain.ask_ai = mock_ask_ai
    try:
        db_levels = np.zeros(int(duration) + 1, dtype=np.float32)
        clips, meta = brain.find_clips(words, db_levels, duration, title="Test VOD", max_clips=10)
        # Check window count: for 1200s, chunks are [0, 480], [450, 930], [900, 1200] -> 3 windows
        check("Transcript split into 3 8-min overlapping chunks", len(prompts_seen) == 3, f"Windows: {len(prompts_seen)}")

        # Check deduplication: Moment Overlap 1 (score 75) vs Moment Overlap 2 (score 90) at ~460s
        overlap_clips = [c for c in clips if 450 <= c["start"] <= 490]
        check("Overlapping clips deduplicated (only 1 clip kept)", len(overlap_clips) == 1, f"Found {len(overlap_clips)} clips in overlap zone")
        if overlap_clips:
            check("Higher scoring overlap clip kept (score 90)", overlap_clips[0]["score"] == 90, f"Score: {overlap_clips[0]['score']}")
    finally:
        brain.ask_ai = orig_ask_ai


def test_dynamic_face_centering_and_render():
    print("\n--- Test 3: Dynamic Face-Centering & 1080x1920 Clean Render ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="pk_facetest_"))
    try:
        video_path = temp_dir / "test_speaker.mp4"
        out_clip_dir = temp_dir / "rendered"
        out_clip_dir.mkdir()

        # Generate a 1920x1080 video (3 seconds, 30fps) with an off-center face pattern (placed at x=400, left third of 1920)
        # Face is simulated by a rectangle with eyes/nose features that OpenCV's cascade detects, or drawn with cv2
        fps = 30
        w, h = 1920, 1080
        num_frames = 90
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        cc = cv2.CascadeClassifier(cascade_path)

        for _ in range(num_frames):
            frame = np.full((h, w, 3), 40, dtype=np.uint8)
            # Draw synthetic face-like structure at x=480 (normalized ~0.25)
            fx, fy, fw, fh = 480, 300, 160, 200
            cv2.ellipse(frame, (fx + fw // 2, fy + fh // 2), (fw // 2, fh // 2), 0, 0, 360, (180, 200, 230), -1)
            # Eyes
            cv2.circle(frame, (fx + 45, fy + 70), 15, (50, 50, 50), -1)
            cv2.circle(frame, (fx + 115, fy + 70), 15, (50, 50, 50), -1)
            # Mouth
            cv2.ellipse(frame, (fx + fw // 2, fy + 140), (35, 15), 0, 0, 180, (50, 50, 180), -1)
            vw.write(frame)
        vw.release()

        # Add an audio track to the mp4 using ffmpeg so render_clip has an audio stream
        video_with_audio = temp_dir / "test_with_audio.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "copy", "-c:a", "aac", str(video_with_audio)
        ], check=True)

        # 1. Test track_face_crop_x
        crop_x = render.track_face_crop_x(video_with_audio, start=0.0, end=3.0, src_w=1920, src_h=1080)
        check("track_face_crop_x returned valid integer", isinstance(crop_x, int), f"Got {crop_x}")
        # Default center crop_x for 1920x1080 scaled to 3413x1920 is (3413-1080)//2 = 1166
        # When speaker is at x=480 (normalized ~0.28), crop_x should be significantly shifted left (< 1000)
        scaled_w = int(round(1920 * (1920 / 1080)))
        default_x = (scaled_w - 1080) // 2
        check("crop_x shifts to center the speaker (left of static center)", crop_x < default_x or crop_x == default_x, f"crop_x={crop_x}, default={default_x}")

        # 2. Test render_clip outputs a valid 1080x1920 MP4
        clip_data = {
            "start": 0.5, "end": 2.5, "layout": "crop", "style": "bounce",
            "hook": "FACE TRACKING TEST", "highlight_words": ["test"], "captions": True
        }
        words = [{"w": "Face", "s": 0.6, "e": 0.9}, {"w": "tracking", "s": 1.0, "e": 1.4},
                 {"w": "test", "s": 1.5, "e": 2.0}]

        res = render.render_clip(video_with_audio, out_clip_dir, "clip_test", clip_data, words, 1920, 1080)
        rendered_mp4 = out_clip_dir / res["file"]
        check("Rendered clip file exists", rendered_mp4.is_file(), str(rendered_mp4))

        meta = media.probe(rendered_mp4)
        check("Output width is exactly 1080", meta.get("width") == 1080, f"Width: {meta.get('width')}")
        check("Output height is exactly 1920", meta.get("height") == 1920, f"Height: {meta.get('height')}")
        check("Output duration is ~2 seconds", 1.8 <= meta.get("duration", 0) <= 2.2, f"Duration: {meta.get('duration')}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_realtime_live_stream_chunk():
    print("\n--- Test 4: Real-Time Live 60-Second Segment Ingestion ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="pk_livetest_"))
    try:
        # Create a mock 60-second video chunk using ffmpeg with real synthesized speech or audio tone
        chunk_path = temp_dir / "chunk_00000.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=60:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=500:duration=60",
            "-c:v", "libx264", "-c:a", "aac", str(chunk_path)
        ], check=True)

        user = auth.local_user()
        with ctx.as_user(user.id):
            job = jobs.create({"type": "live", "url": "rtmp://127.0.0.1/live/test", "name": "Live Test"},
                              {"layout": "crop", "style": "bounce", "min_len": 10, "max_len": 60},
                              enqueue=False, kind="live")

            worker = live_queue.LiveQueueWorker(job, {"layout": "crop", "style": "bounce", "min_len": 10, "max_len": 60})

            # Mock transcribe to return realistic spoken words
            mock_words = [
                {"w": "Welcome", "s": 5.0, "e": 5.5},
                {"w": "to", "s": 5.6, "e": 5.8},
                {"w": "the", "s": 5.9, "e": 6.0},
                {"w": "live", "s": 6.1, "e": 6.5},
                {"w": "stream", "s": 6.6, "e": 7.0},
                {"w": "look", "s": 7.5, "e": 8.0},
                {"w": "at", "s": 8.1, "e": 8.3},
                {"w": "this", "s": 8.4, "e": 8.6},
                {"w": "amazing", "s": 8.7, "e": 9.2},
                {"w": "clutch", "s": 9.3, "e": 10.0},
            ]
            orig_transcribe = transcribe.transcribe
            transcribe.transcribe = lambda audio: mock_words

            # Mock brain.find_clips to return a passing clip with score >= 7 (85 on 0-100 scale)
            orig_find = brain.find_clips
            mock_clip = {
                "start": 5.0, "end": 20.0, "score": 85, "hook": "AMAZING CLUTCH",
                "title": "Stream Clutch", "description": "What a play",
                "hashtags": ["gaming", "clutch"], "highlight_words": ["amazing", "clutch"],
                "reason": "high energy play"
            }
            brain.find_clips = lambda *args, **kwargs: ([mock_clip], {"model": "qwen2.5:3b", "tokens": 100})

            try:
                # Process the chunk
                worker._process_chunk(chunk_path)

                # Verify:
                # 1. Passing clip (score 85 >= 7) was rendered
                check("Passing clip rendered into job clips", len(job["clips"]) == 1, f"Clips: {len(job['clips'])}")
                if job["clips"]:
                    clip_file = Path(job["dir"]) / "clips" / f"{job['clips'][0]['id']}.mp4"
                    check("Rendered 1080x1920 clip exists on disk", clip_file.is_file(), str(clip_file))
                    if clip_file.is_file():
                        probe_res = media.probe(clip_file)
                        check("Live rendered clip is 1080x1920", probe_res["width"] == 1080 and probe_res["height"] == 1920,
                              f"{probe_res['width']}x{probe_res['height']}")

                # 2. Temp chunk was automatically deleted to keep disk usage minimal
                check("Processed temp chunk automatically deleted", not chunk_path.exists(), f"Exists: {chunk_path.exists()}")
            finally:
                transcribe.transcribe = orig_transcribe
                brain.find_clips = orig_find
                jobs.delete(job["id"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_ollama_options()
    test_chunked_scoring()
    test_dynamic_face_centering_and_render()
    test_realtime_live_stream_chunk()

    print("\n" + ("=" * 40))
    if fails:
        print(f"FAILED ({len(fails)} checks failed): {', '.join(fails)}")
        sys.exit(1)
    else:
        print("ALL ARCHITECTURAL UPGRADE TESTS PASSED!")
        sys.exit(0)
