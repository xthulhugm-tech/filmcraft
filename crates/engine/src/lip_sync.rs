//! Automatic lip sync for composite character parts (`clip.lipSync`).
//!
//! Analyzes audio clips for speech activity (RMS/dBFS threshold with debounce smoothing)
//! and generates Hold opacity keyframes on open-mouth and closed-mouth visual clips.

use filmcraft_project::{ClipId, Interpolation, Keyframe, ParamValue, TrackItem, TrackKind, find_effect};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, f64_p, has_seq, u64_p};
use crate::{EngineError, Result, Session};

/// Sample rate used for speech analysis (16 kHz is standard and fast).
const ANALYSIS_SAMPLE_RATE: u32 = 16_000;

/// Default speech energy threshold in dBFS.
pub const DEFAULT_THRESHOLD_DB: f64 = -35.0;

/// Default minimum speech duration in milliseconds to register a voiced segment.
pub const DEFAULT_MIN_SPEECH_MS: f64 = 60.0;

/// Default minimum silence duration in milliseconds to bridge pauses.
pub const DEFAULT_MIN_SILENCE_MS: f64 = 100.0;

/// Default lip-sync mouth opening/closing cycle in milliseconds (e.g. 240ms = 120ms open / 120ms closed).
pub const DEFAULT_CYCLE_MS: f64 = 240.0;

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "clip.lipSync",
        label: "Lip Sync…",
        menu: &["Clip"],
        shortcut: None,
        params: r#"{"audioClip":id?,"openMouthClip":id,"closedMouthClip":id,"thresholdDb":f64=-35.0,"sensitivity":0..100=50.0,"minSpeechMs":f64=60.0,"minSilenceMs":f64=100.0,"cycleMs":f64=240.0}"#,
        enabled: can_lip_sync,
        run: lip_sync,
        journal: true,
    }]
}

fn can_lip_sync(s: &Session) -> std::result::Result<(), String> {
    has_seq(s)?;
    Ok(())
}

/// Options for audio analysis and keyframe generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipSyncOptions {
    pub threshold_db: f64,
    pub min_speech_ms: f64,
    pub min_silence_ms: f64,
    pub cycle_ms: f64,
}

impl Default for LipSyncOptions {
    fn default() -> Self {
        Self {
            threshold_db: DEFAULT_THRESHOLD_DB,
            min_speech_ms: DEFAULT_MIN_SPEECH_MS,
            min_silence_ms: DEFAULT_MIN_SILENCE_MS,
            cycle_ms: DEFAULT_CYCLE_MS,
        }
    }
}

/// Analyze downmixed mono audio samples and return voiced speech segments as `(start_tick, end_tick)`
/// relative to the beginning of the analyzed audio slice.
pub fn analyze_speech_segments(
    audio: &[f32],
    sample_rate: u32,
    opts: &LipSyncOptions,
) -> Vec<(Tick, Tick)> {
    if audio.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    // 10 ms frame window
    let frame_len = (sample_rate as usize / 100).max(1);
    let frame_ticks = (TICKS_PER_SECOND / 100).max(1);

    let num_frames = (audio.len() + frame_len - 1) / frame_len;
    let mut voiced_frames = vec![false; num_frames];

    for (f_idx, chunk) in audio.chunks(frame_len).enumerate() {
        if chunk.is_empty() {
            continue;
        }
        let energy: f32 = chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32;
        let db = if energy > 1e-12 {
            10.0 * (energy as f64).log10()
        } else {
            -120.0
        };
        voiced_frames[f_idx] = db >= opts.threshold_db;
    }

    // Smoothing pass 1: bridge short silences (< min_silence_ms) between speech
    let min_silence_frames = (opts.min_silence_ms / 10.0).round().max(1.0) as usize;
    let mut i = 0;
    while i < voiced_frames.len() {
        if !voiced_frames[i] {
            let start = i;
            while i < voiced_frames.len() && !voiced_frames[i] {
                i += 1;
            }
            let silence_len = i - start;
            // Bridge if preceded and succeeded by voiced frames
            if start > 0 && i < voiced_frames.len() && silence_len < min_silence_frames {
                for f in start..i {
                    voiced_frames[f] = true;
                }
            }
        } else {
            i += 1;
        }
    }

    // Smoothing pass 2: remove short blips (< min_speech_ms)
    let min_speech_frames = (opts.min_speech_ms / 10.0).round().max(1.0) as usize;
    i = 0;
    while i < voiced_frames.len() {
        if voiced_frames[i] {
            let start = i;
            while i < voiced_frames.len() && voiced_frames[i] {
                i += 1;
            }
            let speech_len = i - start;
            if speech_len < min_speech_frames {
                for f in start..i {
                    voiced_frames[f] = false;
                }
            }
        } else {
            i += 1;
        }
    }

    // Convert contiguous voiced frames into (start_tick, end_tick) segments
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut seg_start_frame = 0;

    for (idx, &is_v) in voiced_frames.iter().enumerate() {
        if is_v && !in_speech {
            in_speech = true;
            seg_start_frame = idx;
        } else if !is_v && in_speech {
            in_speech = false;
            let s_tick = Tick(seg_start_frame as i64 * frame_ticks);
            let e_tick = Tick(idx as i64 * frame_ticks);
            segments.push((s_tick, e_tick));
        }
    }

    if in_speech {
        let s_tick = Tick(seg_start_frame as i64 * frame_ticks);
        let e_tick = Tick(voiced_frames.len() as i64 * frame_ticks);
        segments.push((s_tick, e_tick));
    }

    segments
}

/// Helper to ensure the "opacity" effect exists on a track item and returns a mutable reference to its parameter.
fn ensure_opacity_param(item: &mut TrackItem) -> Option<&mut filmcraft_project::Param> {
    if !item.effects.iter().any(|e| e.effect == "opacity") {
        if let Some(def) = find_effect("opacity") {
            item.effects.push(def.instance());
        }
    }
    let eff = item.effects.iter_mut().find(|e| e.effect == "opacity")?;
    eff.params.get_mut("opacity")
}

/// Check if a timeline time is within any of the voiced segments (timeline absolute time).
#[allow(dead_code)]
fn is_time_voiced(t: Tick, tl_segments: &[(Tick, Tick)]) -> bool {
    tl_segments.iter().any(|&(s, e)| t >= s && t < e)
}

/// Generate Hold opacity keyframes on open and closed mouth clips given timeline-absolute voiced segments.
/// Inside each voiced segment, mouth opens and closes periodically according to `cycle_ms`.
pub fn generate_mouth_keyframes(
    mouth_clip: &mut TrackItem,
    is_open_mouth: bool,
    tl_segments: &[(Tick, Tick)],
    cycle_ms: f64,
) -> Result<()> {
    let clip_start_tl = mouth_clip.start;
    let clip_dur = mouth_clip.duration;
    let clip_end_tl = clip_start_tl + clip_dur;

    // Half cycle in ticks (duration of one open state or one closed state)
    let half_cycle_sec = (cycle_ms / 1000.0 / 2.0).max(0.02);
    let half_cycle_ticks = (half_cycle_sec * TICKS_PER_SECOND as f64).round() as i64;

    // Build timeline events: (timeline_tick, is_open)
    let mut events: Vec<(Tick, bool)> = Vec::new();

    // Initial state at clip start is closed (is_open = false)
    events.push((clip_start_tl, false));

    for &(seg_s, seg_e) in tl_segments {
        let s = seg_s.max(clip_start_tl);
        let e = seg_e.min(clip_end_tl);
        if s >= e {
            continue;
        }

        // Before speech starts in this segment, ensure mouth is closed
        if s > clip_start_tl {
            events.push((s, true));
        }

        // Oscillate open and closed states across the voiced segment
        let mut cur = s.0;
        let mut open_state = true;
        while cur + half_cycle_ticks <= e.0 {
            cur += half_cycle_ticks;
            open_state = !open_state;
            events.push((Tick(cur), open_state));
        }

        // At speech segment end, close the mouth
        events.push((e, false));
    }

    events.sort_by_key(|(t, _)| t.0);

    let mut keyframes = Vec::new();
    let mut last_val: Option<f64> = None;

    for (pt_tl, open_state) in events {
        if pt_tl < clip_start_tl || pt_tl > clip_end_tl {
            continue;
        }
        let clip_tick = pt_tl - clip_start_tl;

        let target_opacity = if is_open_mouth {
            if open_state { 100.0 } else { 0.0 }
        } else {
            if open_state { 0.0 } else { 100.0 }
        };

        if last_val != Some(target_opacity) || keyframes.is_empty() {
            let mut kf = Keyframe::new(clip_tick, ParamValue::Float(target_opacity));
            kf.interp = Interpolation::Hold;
            keyframes.push(kf);
            last_val = Some(target_opacity);
        }
    }

    let param = ensure_opacity_param(mouth_clip)
        .ok_or_else(|| EngineError::Other("failed to resolve opacity parameter".into()))?;

    // Apply generated keyframes
    param.keyframes = keyframes;
    if let Some(first) = param.keyframes.first() {
        param.value = first.value.clone();
    }

    Ok(())
}

fn lip_sync(s: &mut Session, p: &Value) -> Result<Value> {
    let _seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;

    // 1. Resolve audio clip
    let audio_clip_id = u64_p(p, "audioClip").map(ClipId).or_else(|| {
        let q = s.active_sequence()?;
        s.state.selection.iter().copied().find(|&c| {
            q.find_item(c).is_some_and(|(t, _)| {
                q.track(t).is_some_and(|tr| tr.kind == TrackKind::Audio)
            })
        })
    }).ok_or_else(|| bad("clip.lipSync", "specify `audioClip` or select an audio clip in the timeline"))?;

    // 2. Resolve open and closed mouth clips
    let (auto_open, auto_closed) = {
        let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let v_clips: Vec<(ClipId, String)> = s.state.selection.iter().filter_map(|&c| {
            let (tr_id, item) = q.find_item(c)?;
            if q.track(tr_id).is_some_and(|tr| tr.kind == TrackKind::Video) {
                let item_name = s.project.item(item.item).map(|it| it.name.to_lowercase()).unwrap_or_default();
                Some((c, item_name))
            } else {
                None
            }
        }).collect();

        let open = v_clips.iter().find(|(_, name)| name.contains("open") || name.contains("開")).map(|(c, _)| *c)
            .or_else(|| v_clips.first().map(|(c, _)| *c));
        let closed = v_clips.iter().find(|(c, name)| (name.contains("close") || name.contains("閉")) && Some(*c) != open).map(|(c, _)| *c)
            .or_else(|| v_clips.iter().map(|(c, _)| *c).find(|&c| Some(c) != open));
        (open, closed)
    };

    let open_clip_id = u64_p(p, "openMouthClip").map(ClipId).or(auto_open).ok_or_else(|| {
        bad("clip.lipSync", "`openMouthClip` is required (specify `openMouthClip` or select mouth video clips)")
    })?;
    let closed_clip_id = u64_p(p, "closedMouthClip").map(ClipId).or(auto_closed).ok_or_else(|| {
        bad("clip.lipSync", "`closedMouthClip` is required (specify `closedMouthClip` or select mouth video clips)")
    })?;

    if open_clip_id == closed_clip_id {
        return Err(bad("clip.lipSync", "`openMouthClip` and `closedMouthClip` must be different clips"));
    }

    // 3. Resolve options
    let threshold_db = f64_p(p, "thresholdDb").unwrap_or_else(|| {
        let sensitivity = f64_p(p, "sensitivity").unwrap_or(50.0).clamp(0.0, 100.0);
        // Sensitivity 50 -> -35 dBFS; 0 -> -20 dBFS; 100 -> -50 dBFS
        -20.0 - (sensitivity / 100.0) * 30.0
    });
    let min_speech_ms = f64_p(p, "minSpeechMs").unwrap_or(DEFAULT_MIN_SPEECH_MS).max(10.0);
    let min_silence_ms = f64_p(p, "minSilenceMs").unwrap_or(DEFAULT_MIN_SILENCE_MS).max(10.0);
    let cycle_ms = f64_p(p, "cycleMs").unwrap_or(DEFAULT_CYCLE_MS).max(40.0);

    let opts = LipSyncOptions {
        threshold_db,
        min_speech_ms,
        min_silence_ms,
        cycle_ms,
    };

    // 4. Retrieve audio samples from session media
    let (audio_item_id, audio_start_tl, audio_dur_tl, audio_source_in) = {
        let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let (_, clip) = q.find_item(audio_clip_id).ok_or_else(|| bad("clip.lipSync", "audio clip not found in active sequence"))?;
        (clip.item, clip.start, clip.duration, clip.source_in)
    };

    let (media_root_item, _, _) = s.project.resolve_media(audio_item_id)
        .ok_or_else(|| bad("clip.lipSync", "failed to resolve audio clip media item"))?;

    let src = s.source(media_root_item)
        .ok_or_else(|| bad("clip.lipSync", "audio source media is offline or inaccessible"))?;

    let sr = ANALYSIS_SAMPLE_RATE;
    let start_sample = (audio_source_in.0 as f64 * sr as f64 / TICKS_PER_SECOND as f64).round().max(0.0) as i64;
    let len_samples = (audio_dur_tl.0 as f64 * sr as f64 / TICKS_PER_SECOND as f64).round().max(0.0) as usize;

    if len_samples == 0 {
        return Err(bad("clip.lipSync", "audio clip duration is zero"));
    }

    let buf = src.audio(start_sample, len_samples, sr)
        .map_err(|e| EngineError::Other(format!("failed to decode audio samples: {e}")))?;

    let mono_samples = filmcraft_speech::downmix(&buf.channels);

    // 5. Analyze speech segments relative to audio clip start
    let voiced_clip_rel = analyze_speech_segments(&mono_samples, sr, &opts);

    // Convert to timeline absolute segments
    let tl_segments: Vec<(Tick, Tick)> = voiced_clip_rel
        .into_iter()
        .map(|(s_tick, e_tick)| (audio_start_tl + s_tick, audio_start_tl + e_tick))
        .collect();

    let num_segments = tl_segments.len();

    // 6. Apply keyframes to mouth clips inside an atomic undoable edit
    s.edit_sequence("Auto Lip Sync", move |seq, _ctx, _st| {
        // Verify both mouth clips exist in video tracks
        let open_exists = seq.find_item(open_clip_id).is_some_and(|(t, _)| {
            seq.track(t).is_some_and(|tr| tr.kind == TrackKind::Video)
        });
        if !open_exists {
            return Err(bad("clip.lipSync", "open mouth clip not found in sequence video tracks"));
        }

        let closed_exists = seq.find_item(closed_clip_id).is_some_and(|(t, _)| {
            seq.track(t).is_some_and(|tr| tr.kind == TrackKind::Video)
        });
        if !closed_exists {
            return Err(bad("clip.lipSync", "closed mouth clip not found in sequence video tracks"));
        }

        // Apply to open mouth
        let (_, open_clip) = seq.find_item_mut(open_clip_id)
            .ok_or_else(|| bad("clip.lipSync", "open mouth clip not found"))?;
        generate_mouth_keyframes(open_clip, true, &tl_segments, opts.cycle_ms)?;

        // Apply to closed mouth
        let (_, closed_clip) = seq.find_item_mut(closed_clip_id)
            .ok_or_else(|| bad("clip.lipSync", "closed mouth clip not found"))?;
        generate_mouth_keyframes(closed_clip, false, &tl_segments, opts.cycle_ms)?;

        Ok(())
    })?;

    Ok(json!({
        "audioClip": audio_clip_id.0,
        "openMouthClip": open_clip_id.0,
        "closedMouthClip": closed_clip_id.0,
        "speechSegmentsDetected": num_segments,
        "thresholdDb": opts.threshold_db,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_media::generators::GeneratorSource;
    use filmcraft_project::{ItemId, ItemKind};
    use filmcraft_media::{DemoScene, Generator, MediaSource};
    use filmcraft_project::{Label, MediaClip, MediaRef, Project, SequenceSettings};
    use filmcraft_time::{FrameRate, TimeRange};
    use std::path::Path;

    #[test]
    fn test_analyze_pure_silence() {
        let silence = vec![0.0f32; 16000]; // 1 second of digital silence
        let opts = LipSyncOptions::default();
        let segments = analyze_speech_segments(&silence, 16000, &opts);
        assert!(segments.is_empty(), "silence should produce zero voiced segments");
    }

    #[test]
    fn test_analyze_continuous_speech() {
        // 1 second of loud sine wave (approx -6 dBFS)
        let sine: Vec<f32> = (0..16000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let opts = LipSyncOptions::default();
        let segments = analyze_speech_segments(&sine, 16000, &opts);
        assert_eq!(segments.len(), 1, "continuous tone should produce exactly 1 voiced segment");
        assert_eq!(segments[0].0, Tick(0));
        assert!(segments[0].1.0 >= TICKS_PER_SECOND - TICKS_PER_SECOND / 100);
    }

    #[test]
    fn test_analyze_alternating_speech_and_silence() {
        // Pattern: 0.0 - 0.4s Speech, 0.4 - 0.7s Silence, 0.7 - 1.0s Speech
        let mut audio = vec![0.0f32; 16000];
        for i in 0..6400 {
            audio[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
        }
        for i in 11200..16000 {
            audio[i] = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
        }

        let opts = LipSyncOptions::default();
        let segments = analyze_speech_segments(&audio, 16000, &opts);
        assert_eq!(segments.len(), 2, "should detect two separate speech segments: {segments:?}");

        // Segment 1 approx 0 to 0.4s
        assert_eq!(segments[0].0, Tick(0));
        let seg1_end_sec = segments[0].1.0 as f64 / TICKS_PER_SECOND as f64;
        assert!((seg1_end_sec - 0.4).abs() < 0.03, "segment 1 end was {seg1_end_sec}");

        // Segment 2 approx 0.7s to 1.0s
        let seg2_start_sec = segments[1].0.0 as f64 / TICKS_PER_SECOND as f64;
        assert!((seg2_start_sec - 0.7).abs() < 0.03, "segment 2 start was {seg2_start_sec}");
    }

    #[test]
    fn test_lip_sync_keyframes_timing_and_hold() {
        let rate = FrameRate::FPS_30;
        let mut p = Project::new("lip_sync_test");
        let _seq = p.new_sequence("seq", SequenceSettings { width: 1920, height: 1080, frame_rate: rate, ..Default::default() }, 2, 1, None);

        // Create dummy video items for open and closed mouth
        let src = GeneratorSource::new(Generator::Demo(DemoScene::OceanSunset), 1920, 1080, rate, rate.tick_of(300));
        let clip_meta = MediaClip {
            media: MediaRef::Generator(Generator::Demo(DemoScene::OceanSunset)),
            info: src.info().clone(),
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        };

        let open_item = p.add_item("mouth_open", Label::Iris, ItemKind::Media(clip_meta.clone()), None);
        let closed_item = p.add_item("mouth_closed", Label::Iris, ItemKind::Media(clip_meta), None);

        // Place both clips from t=1.0s to t=3.0s (duration = 2.0s = 60 frames)
        let clip_start = rate.tick_of(30);
        let clip_dur = rate.tick_of(60);

        let mut v_open = p.make_track_item(open_item, TrackKind::Video, clip_start, TimeRange::new(Tick::ZERO, clip_dur), rate).unwrap();
        let mut v_closed = p.make_track_item(closed_item, TrackKind::Video, clip_start, TimeRange::new(Tick::ZERO, clip_dur), rate).unwrap();

        // Voiced segment in timeline time: t=1.5s to t=2.5s (inside the 1.0..3.0s clip)
        let tl_segments = vec![(rate.tick_of(45), rate.tick_of(75))];

        generate_mouth_keyframes(&mut v_open, true, &tl_segments, 200.0).unwrap();
        generate_mouth_keyframes(&mut v_closed, false, &tl_segments, 200.0).unwrap();

        let open_eff = v_open.effects.iter().find(|e| e.effect == "opacity").unwrap();
        let open_kfs = &open_eff.params["opacity"].keyframes;

        let closed_eff = v_closed.effects.iter().find(|e| e.effect == "opacity").unwrap();
        let closed_kfs = &closed_eff.params["opacity"].keyframes;

        // Verify keyframes on Open Mouth:
        // Keyframes oscillate between 100.0 and 0.0 during voiced segment
        assert!(open_kfs.len() > 3, "Expected multiple oscillating keyframes, got {}", open_kfs.len());
        assert_eq!(open_kfs[0].time, Tick::ZERO);
        assert_eq!(open_kfs[0].value, ParamValue::Float(0.0));
        assert_eq!(open_kfs[0].interp, Interpolation::Hold);

        assert_eq!(open_kfs[1].time, rate.tick_of(15));
        assert_eq!(open_kfs[1].value, ParamValue::Float(100.0));
        assert_eq!(open_kfs[1].interp, Interpolation::Hold);

        // Verify mutual exclusivity between open and closed mouth keyframes
        assert_eq!(open_kfs.len(), closed_kfs.len());
        for (okf, ckf) in open_kfs.iter().zip(closed_kfs.iter()) {
            assert_eq!(okf.time, ckf.time);
            assert_eq!(okf.interp, Interpolation::Hold);
            assert_eq!(ckf.interp, Interpolation::Hold);
            let oval = match okf.value { ParamValue::Float(v) => v, _ => panic!() };
            let cval = match ckf.value { ParamValue::Float(v) => v, _ => panic!() };
            assert_eq!(oval + cval, 100.0, "Values must sum to 100%");
        }
    }

    fn write_test_wav(path: &Path, samples: &[f32], sample_rate: u32) {
        let mut data = Vec::new();
        let num_samples = samples.len() as u32;
        let byte_rate = sample_rate * 2;
        let data_len = num_samples * 2;

        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36 + data_len).to_le_bytes());
        data.extend_from_slice(b"WAVE");

        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes()); // PCM
        data.extend_from_slice(&1u16.to_le_bytes()); // mono
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&byte_rate.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes()); // block align
        data.extend_from_slice(&16u16.to_le_bytes()); // 16-bit

        data.extend_from_slice(b"data");
        data.extend_from_slice(&data_len.to_le_bytes());
        for &s in samples {
            let sample_i16 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            data.extend_from_slice(&sample_i16.to_le_bytes());
        }

        std::fs::write(path, data).unwrap();
    }

    #[test]
    fn test_lip_sync_session_command_undo_redo_and_persistence() {
        let dir = crate::media_test_util::tmp_dir("lipsync-e2e");
        let wav_path = dir.join("test_speech.wav");

        // 2 seconds of audio: 0.0..0.8s Speech, 0.8..1.2s Silence, 1.2..2.0s Speech
        let mut samples = vec![0.0f32; 32000];
        for i in 0..12800 {
            samples[i] = 0.6 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
        }
        for i in 19200..32000 {
            samples[i] = 0.6 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
        }
        write_test_wav(&wav_path, &samples, 16000);

        let mut s = Session::default();
        let r = s.execute("file.import", json!({"paths": [wav_path.to_str().unwrap()]})).unwrap();
        let audio_item_id = ItemId(r["items"][0].as_u64().unwrap());

        let rate = FrameRate::FPS_30;
        let mut p = (*s.project).clone();
        let seq = p.new_sequence("Seq", SequenceSettings { width: 1920, height: 1080, frame_rate: rate, ..Default::default() }, 2, 1, None);

        // Mouth items
        let mouth_src = GeneratorSource::new(Generator::Demo(DemoScene::OceanSunset), 1920, 1080, rate, rate.tick_of(300));
        let clip_meta = MediaClip {
            media: MediaRef::Generator(Generator::Demo(DemoScene::OceanSunset)),
            info: mouth_src.info().clone(),
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        };
        let open_item = p.add_item("open", Label::Iris, ItemKind::Media(clip_meta.clone()), None);
        let closed_item = p.add_item("closed", Label::Iris, ItemKind::Media(clip_meta), None);

        // Place on timeline:
        // Audio at t=1.0s (rate.tick_of(30)), duration=2.0s (rate.tick_of(60)), trimmed source_in=0.2s (rate.tick_of(6))
        let audio_start = rate.tick_of(30);
        let audio_dur = rate.tick_of(60);
        let audio_source_in = rate.tick_of(6);

        let audio_track_item = p.make_track_item(audio_item_id, TrackKind::Audio, audio_start, TimeRange::new(audio_source_in, audio_dur), rate).unwrap();
        let audio_clip_id = audio_track_item.id;

        // Mouth items on V1 and V2, covering t=0.5s to t=3.5s
        let mouth_start = rate.tick_of(15);
        let mouth_dur = rate.tick_of(90);
        let open_track_item = p.make_track_item(open_item, TrackKind::Video, mouth_start, TimeRange::new(Tick::ZERO, mouth_dur), rate).unwrap();
        let closed_track_item = p.make_track_item(closed_item, TrackKind::Video, mouth_start, TimeRange::new(Tick::ZERO, mouth_dur), rate).unwrap();
        let open_clip_id = open_track_item.id;
        let closed_clip_id = closed_track_item.id;

        {
            let q = p.sequence_mut(seq).unwrap();
            q.audio_tracks[0].items.push(audio_track_item);
            q.video_tracks[0].items.push(open_track_item);
            q.video_tracks[1].items.push(closed_track_item);
        }

        s.project = std::sync::Arc::new(p);
        s.state.active_sequence = Some(seq);
        s.state.open_sequences = vec![seq];

        // Execute lip sync command
        let res = s.execute("clip.lipSync", json!({
            "audioClip": audio_clip_id.0,
            "openMouthClip": open_clip_id.0,
            "closedMouthClip": closed_clip_id.0,
            "thresholdDb": -35.0,
        })).unwrap();

        assert_eq!(res["audioClip"], audio_clip_id.0);
        assert!(res["speechSegmentsDetected"].as_u64().unwrap() >= 1);

        // Verify keyframes generated
        let q = s.active_sequence().unwrap();
        let (_, open_clip) = q.find_item(open_clip_id).unwrap();
        let open_eff = open_clip.effects.iter().find(|e| e.effect == "opacity").unwrap();
        assert!(!open_eff.params["opacity"].keyframes.is_empty(), "open mouth must have keyframes");

        let (_, closed_clip) = q.find_item(closed_clip_id).unwrap();
        let closed_eff = closed_clip.effects.iter().find(|e| e.effect == "opacity").unwrap();
        assert!(!closed_eff.params["opacity"].keyframes.is_empty(), "closed mouth must have keyframes");

        // Verify Undo restores original state (keyframes empty)
        assert_eq!(s.history.undo.last().unwrap().0, "Auto Lip Sync");
        let undo_res = s.undo();
        assert_eq!(undo_res, Some("Auto Lip Sync".into()));

        let q_undo = s.active_sequence().unwrap();
        let (_, open_undo) = q_undo.find_item(open_clip_id).unwrap();
        let open_eff_undo = open_undo.effects.iter().find(|e| e.effect == "opacity").unwrap();
        assert!(open_eff_undo.params["opacity"].keyframes.is_empty(), "keyframes must be cleared on undo");

        // Verify Redo re-applies keyframes
        let redo_res = s.redo();
        assert_eq!(redo_res, Some("Auto Lip Sync".into()));

        let q_redo = s.active_sequence().unwrap();
        let (_, open_redo) = q_redo.find_item(open_clip_id).unwrap();
        let open_eff_redo = open_redo.effects.iter().find(|e| e.effect == "opacity").unwrap();
        assert!(!open_eff_redo.params["opacity"].keyframes.is_empty(), "keyframes must be restored on redo");

        // Verify project persistence: serialize to JSON and deserialize
        let serialized = serde_json::to_string(&*s.project).unwrap();
        let loaded: Project = serde_json::from_str(&serialized).unwrap();
        let q_loaded = loaded.sequence(seq).unwrap();
        let (_, open_loaded) = q_loaded.find_item(open_clip_id).unwrap();
        let open_eff_loaded = open_loaded.effects.iter().find(|e| e.effect == "opacity").unwrap();
        assert_eq!(open_eff_loaded.params["opacity"].keyframes, open_eff_redo.params["opacity"].keyframes, "keyframes must persist across save/load");

        // Test auto-resolve when selecting clips without passing explicit IDs
        s.state.selection = vec![audio_clip_id, open_clip_id, closed_clip_id];
        let auto_res = s.execute("clip.lipSync", json!({})).unwrap();
        assert_eq!(auto_res["audioClip"], audio_clip_id.0);
        assert_eq!(auto_res["openMouthClip"], open_clip_id.0);
        assert_eq!(auto_res["closedMouthClip"], closed_clip_id.0);
    }
}
