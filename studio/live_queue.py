"""Real-time live stream ingestion and rolling chunk processing.

Accepts YouTube, Twitch, Kick, or RTMP URLs, segments rolling 60-second chunks using yt-dlp + ffmpeg,
transcribes each chunk immediately through sherpa-onnx Parakeet, scores with Ollama, renders passing
clips (score >= 7) into 1080x1920 vertical format with dynamic face-centering, and automatically
deletes processed temp chunks to keep disk usage minimal.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import auth, brain, config, ctx, db, jobs, live, media, render, transcribe

LIVE_WORKERS: dict = {}
_LOCK = threading.Lock()


def resolve_stream(url: str) -> dict:
    """Resolve a stream URL (YouTube, Twitch, Kick, or RTMP) to playable media URLs."""
    url = url.strip()
    if url.startswith(("rtmp://", "rtmps://")):
        return {"urls": [url], "title": "RTMP Stream", "uploader": "Live Stream"}
    try:
        return live.resolve(url)
    except Exception:
        # Fallback if yt-dlp direct resolve fails (e.g. direct HLS or mock stream)
        return {"urls": [url], "title": "Live Stream", "uploader": "Live"}


class LiveQueueWorker:
    def __init__(self, job: dict, options: dict = None):
        self.job = job
        self.options = options or {}
        self.stop_event = threading.Event()
        self.job_dir = Path(job["dir"])
        self.chunks_dir = self.job_dir / "live_chunks"
        self.clips_dir = self.job_dir / "clips"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.proc = None
        self.thread = None
        self.all_words = []
        self.processed_chunks = set()

    def start(self):
        self.thread = threading.Thread(target=ctx.bind(self._run), daemon=True, name=f"live_worker_{self.job['id']}")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _run(self):
        url = self.job["source"]["url"]
        try:
            info = resolve_stream(url)
            self.job.update(title=info.get("title") or self.job.get("title") or "Live Stream",
                            uploader=info.get("uploader") or self.job.get("uploader") or "Live")
            self.job.update(status="live", stage="Ingesting live stream")
            jobs.save(self.job)
            ctx.bump()

            stream_urls = info.get("urls") or [url]
            seg_csv = self.chunks_dir / "segments.csv"

            # Rolling 60-second chunks using ffmpeg
            cmd = ["ffmpeg", "-y", "-v", "error", "-reconnect", "1", "-reconnect_streamed", "1",
                   "-reconnect_delay_max", "10"]
            for u in stream_urls:
                cmd += ["-i", u]
            a_in = "1:a:0" if len(stream_urls) > 1 else "0:a:0"
            cmd += ["-map", "0:v:0", "-map", a_in, "-c", "copy",
                    "-f", "segment", "-segment_time", "60", "-reset_timestamps", "1",
                    "-segment_list", str(seg_csv), "-segment_list_type", "csv",
                    str(self.chunks_dir / "chunk_%05d.mp4")]

            with open(self.job_dir / "ffmpeg_live.log", "ab") as errlog:
                self.proc = subprocess.Popen(cmd, cwd=self.chunks_dir, stdout=subprocess.DEVNULL, stderr=errlog,
                                             creationflags=media.NO_WINDOW)

            csv_pos = 0
            while not self.stop_event.is_set():
                if self.proc and self.proc.poll() is not None:
                    # ffmpeg ended
                    break

                if seg_csv.is_file():
                    try:
                        content = seg_csv.read_text(encoding="utf-8", errors="ignore")
                        lines = [line.strip() for line in content.splitlines() if line.strip()]
                    except Exception:
                        lines = []

                    while csv_pos < len(lines):
                        line = lines[csv_pos]
                        csv_pos += 1
                        parts = line.split(",")
                        if len(parts) >= 1:
                            chunk_file = self.chunks_dir / parts[0].strip()
                            self._process_chunk(chunk_file)

                time.sleep(1)

            # Process any remaining lines after ffmpeg exit
            if seg_csv.is_file():
                try:
                    content = seg_csv.read_text(encoding="utf-8", errors="ignore")
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    while csv_pos < len(lines):
                        chunk_file = self.chunks_dir / lines[csv_pos].split(",")[0].strip()
                        csv_pos += 1
                        self._process_chunk(chunk_file)
                except Exception:
                    pass

        except Exception as e:
            self.job["error"] = f"Live ingestion error: {e}"
        finally:
            self._cleanup()

    def _process_chunk(self, chunk_path: Path):
        """Process a completed 60-second chunk immediately through sherpa-onnx + Ollama scorer."""
        if not chunk_path.is_file() or str(chunk_path) in self.processed_chunks:
            return
        self.processed_chunks.add(str(chunk_path))

        tmp_wav = chunk_path.with_suffix(".wav")
        try:
            # 1. Audio extraction
            media.extract_audio(chunk_path, tmp_wav)
            if not tmp_wav.is_file() or tmp_wav.stat().st_size < 1000:
                return
            audio = media.load_audio(tmp_wav)
        except Exception:
            return
        finally:
            tmp_wav.unlink(missing_ok=True)

        try:
            # 2. sherpa-onnx Parakeet transcription
            words = transcribe.transcribe(audio)
            if not words or len(words) < 3:
                return

            self.all_words.extend(words)
            (self.job_dir / "words.json").write_text(jobs.json.dumps(self.all_words), encoding="utf-8")

            # 3. Ollama scoring
            duration = max(1.0, len(audio) / 16000.0)
            db_levels = media.loudness_per_second(audio)
            clips, _ai_meta = brain.find_clips(
                words=words,
                db=db_levels,
                duration=duration,
                title=self.job.get("title", ""),
                uploader=self.job.get("uploader", ""),
                max_clips=2,
                min_len=int(self.options.get("min_len", 10)),
                max_len=int(self.options.get("max_len", 60)),
                game=self.job.get("game"),
            )

            # 4. Render passing clips (score >= 7 on 1-10 scale or >= 70 on 0-100 scale)
            meta = media.probe(chunk_path)
            src_w, src_h = meta["width"], meta["height"]

            for c in clips:
                raw_score = c.get("score", 0)
                is_passing = raw_score >= 70 if raw_score > 10 else raw_score >= 7
                if not is_passing:
                    continue

                clip_idx = len(self.job["clips"]) + 1
                clip_id = f"clip{clip_idx:02d}"
                c.update(
                    id=clip_id,
                    status="queued",
                    layout=self.options.get("layout", "crop"),
                    style=self.options.get("style", "bounce"),
                    captions=self.options.get("captions", True),
                    highlight_words=c.get("highlight_words", []),
                )

                # Render vertical 1080x1920 clip with dynamic face-centering
                render.render_clip(chunk_path, self.clips_dir, clip_id, c, words, src_w, src_h)
                c["status"] = "done"
                c["made_at"] = time.time()
                self.job["clips"].append(c)
                jobs.save(self.job)
                ctx.bump()

        except Exception as e:
            self.job.setdefault("notes", []).append(f"Chunk processing error: {e}")
        finally:
            # 5. Delete processed temp chunk automatically to keep disk usage minimal
            chunk_path.unlink(missing_ok=True)

    def _cleanup(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        shutil.rmtree(self.chunks_dir, ignore_errors=True)
        self.job.update(status="done", stage="Live ingestion finished")
        jobs.save(self.job)
        ctx.bump()
        with _LOCK:
            LIVE_WORKERS.pop(self.job["id"], None)


def start_live_ingestion(url: str, creator_id: str = None, options: dict = None) -> dict:
    """Entry point to create and start a live ingestion worker."""
    url = url.strip()
    if not media.site_allowed(url) and not url.startswith(("rtmp://", "rtmps://")):
        raise ValueError("Invalid live stream URL. Only YouTube, Twitch, Kick, and RTMP URLs are supported.")

    job = jobs.create({"type": "live", "url": url, "name": f"Live: {url[:40]}"},
                      options or {}, enqueue=False, creator=creator_id, kind="live")
    job.update(status="live", stage="Starting live worker")
    jobs.save(job)

    worker = LiveQueueWorker(job, options or {})
    with _LOCK:
        LIVE_WORKERS[job["id"]] = worker
    worker.start()
    return job


def stop_live_ingestion(job_id: str) -> bool:
    with _LOCK:
        worker = LIVE_WORKERS.get(job_id)
    if worker:
        worker.stop()
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeakCut Real-Time Live Stream Ingestion")
    parser.add_argument("--live", required=True, help="Live stream URL (YouTube, Twitch, Kick, RTMP)")
    args = parser.parse_args()

    db.init()
    print(f"Starting real-time live ingestion for {args.live}...")
    user = auth.local_user()
    with ctx.as_user(user.id):
        live_job = start_live_ingestion(args.live)
        print(f"Live job created: {live_job['id']}")
        try:
            while live_job["status"] == "live":
                time.sleep(2)
        except KeyboardInterrupt:
            print("Stopping live ingestion...")
            stop_live_ingestion(live_job["id"])
