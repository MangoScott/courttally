# Recording for CourtTally

The recording is the product. Nothing downstream can add back what the capture
lost, so the pipeline treats the original file as read-only and cuts from it with
stream copy (no re-encode) by default.

## Phone recipe (today, native camera app)

- Tripod or fence mount behind the baseline, centered, lens about 1.5-2.5 m high,
  tilted so both baselines are in frame with a little margin. Do not zoom.
- 4K at 60 fps if the phone can. 60 fps halves motion blur on serves and gives
  bounce timing to ±8 ms. 1080p60 beats 4K30 for analysis; 4K60 is best.
- HEVC (H.265) "High Efficiency" is fine; CourtTally tone-maps HDR and builds an
  SDR proxy for analysis and review. Turn off "Auto FPS" / "Enhanced
  Stabilization" / cinematic modes: they introduce variable frame rate and crop.
- Lock exposure and focus (tap and hold) once the court is framed. Auto exposure
  hunting during a point looks like motion to the activity detector.
- Keep the microphone uncovered. Ball strikes are the strongest dead-time cue.
- Airplane mode, screen brightness down. A 90 minute 4K60 HEVC session is
  roughly 40 GB; check free space first.

## What `ct ingest` checks

`outputs/<clip>/ingest.json` records codec, resolution, nominal vs average frame
rate (variable frame rate flag), HDR transfer, rotation and audio presence. The
proxy (`proxy.mp4`, 1280 px wide, constant frame rate, H.264) is what analysis
and `review.html` use. Exports always come from the original.

## Capture app (planned)

A first-party iOS capture app is on the roadmap because the native camera app
cannot be told to lock everything we want. Requirements, so the backend does not
have to guess later:

- AVFoundation session: 3840x2160 at 60 fps, HEVC, constant frame rate, 10-bit
  when available; stabilization off; exposure, focus and white balance locked
  after a 2 s auto settle at session start.
- Write to segmented MP4 (fragmented, 1 min segments) so a crash or a full disk
  loses at most a minute, and so upload or on-device analysis can start before
  the session ends.
- Record a sidecar JSON: device model, lens, fps, start wall-clock time, and the
  court keypoints if the user tapped them in the framing screen (feeds
  `ct court --keypoints`).
- Optional: a live "point detector" using the same motion + audio activity
  signal, so the phone can mark points in real time and show the current score
  with a one-tap "who won" prompt between points.
