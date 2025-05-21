# mixer_engine.py
import os
import subprocess
import time
import io
import json # For yt-dlp search
import shutil # For moving files from Spleeter CLI output

import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment
from googleapiclient.discovery import build
from spleeter.separator import Separator # Ensure Spleeter is installed and working
from spleeter.audio.adapter import AudioAdapter

COOKIES_FILE_PATH = os.path.join(os.path.dirname(__file__), "cookies.txt")
USER_AGENT_STRING = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"


YOUTUBE_API_KEY_FROM_ENV = os.getenv('YOUTUBE_API_KEY')
youtube_service = None

def init_youtube_api(api_key_val=None):
    global youtube_service, YOUTUBE_API_KEY_FROM_ENV
    actual_key = api_key_val if api_key_val else YOUTUBE_API_KEY_FROM_ENV
    if not actual_key:
        print("WARNING: mixer_engine - YouTube API Key not provided. API features will fail.")
        return
    try:
        youtube_service = build("youtube", "v3", developerKey=actual_key)
        print("Mixer Engine: YouTube API Initialized.")
    except Exception as e:
        print(f"Mixer Engine: Failed to initialize YouTube API: {e}")
        youtube_service = None


def get_youtube_video_details_via_api(video_id):
    if not youtube_service:
        # print("Mixer Engine: YouTube API not initialized, cannot fetch video details via API.")
        return None
    try:
        req = youtube_service.videos().list(part='snippet,contentDetails', id=video_id).execute()
        if req.get('items'):
            item = req['items'][0]
            title = item['snippet']['title']
            return {"title": title, "id": video_id, "thumbnail_url": item['snippet']['thumbnails']['default']['url']}
    except Exception as e:
        print(f"Mixer Engine: Error fetching API details for {video_id}: {e}")
    return None

def download_youtube_as_mp3(video_id, base_download_dir, use_api_for_title=True):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_specific_dir = os.path.join(base_download_dir, video_id)
    os.makedirs(video_specific_dir, exist_ok=True)
    output_filename = "audio.mp3"
    expected_mp3_path = os.path.join(video_specific_dir, output_filename)
    title_to_return = video_id

    if os.path.exists(expected_mp3_path) and os.path.getsize(expected_mp3_path) > 1024:
        print(f"Mixer Engine: MP3 Cache hit for {video_id}: {expected_mp3_path}")
        if use_api_for_title and youtube_service:
            details = get_youtube_video_details_via_api(video_id)
            if details: title_to_return = details['title']
        return expected_mp3_path, title_to_return

    print(f"Mixer Engine: Downloading {video_url} for {video_id}")
    try:
        yt_dlp_command = [
            'yt-dlp',
            '-x',
            '--audio-format', 'mp3',
            '-f', 'bestaudio/best',
            '--no-playlist', '--no-warnings', #'--no-progress', # Keep progress for verbose logs
            '--progress', # Enable for more detailed output during download
            '--concurrent-fragments', '4', '--retries', '2',
            '--output', expected_mp3_path,
            '--max-filesize', '100M',
            '--max-downloads', '1', # CRITICAL: Ensure this is 1 for single video URLs
            '--user-agent', USER_AGENT_STRING,
        ]

        if os.path.exists(COOKIES_FILE_PATH):
            print(f"Mixer Engine: Using cookies from {COOKIES_FILE_PATH}")
            yt_dlp_command.extend(['--cookies', COOKIES_FILE_PATH])
        # else:
            # print(f"Mixer Engine WARNING: Cookies file not found at {COOKIES_FILE_PATH}.")
        yt_dlp_command.append(video_url)
        result = subprocess.run(
            yt_dlp_command,
            capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=180
        )

        # Robust success check: Prioritize file existence and validity
        if os.path.exists(expected_mp3_path) and os.path.getsize(expected_mp3_path) > 1024: # Min 1KB
            print(f"Mixer Engine: Verified MP3 file exists for {video_id} at {expected_mp3_path}")
            if use_api_for_title and youtube_service:
                details = get_youtube_video_details_via_api(video_id)
                if details: title_to_return = details['title']
            
            if result.returncode != 0:
                # This warning helps understand if yt-dlp exited non-zero despite success due to --max-downloads
                print(f"Mixer Engine WARNING: yt-dlp for {video_id} exited with code {result.returncode}, "
                      f"but MP3 file was created. Assuming success for this one video.")
                # Log the output that might have caused non-zero RC for inspection
                # print(f"   yt-dlp STDOUT (last 200 chars): {result.stdout[-200:]}")
                # print(f"   yt_dlp STDERR (last 200 chars): {result.stderr[-200:]}")
            return expected_mp3_path, title_to_return
        else:
            # File doesn't exist or is too small, so it's a genuine failure.
            error_msg = result.stderr.strip() or result.stdout.strip() or "yt-dlp download error - MP3 not found/empty post-process"
            if "requested format is not available" in error_msg.lower():
                print(f"Mixer Engine: Download for {video_id} failed - format issue. yt-dlp output: {error_msg[:200]}")
            elif "sign in to confirm" in error_msg.lower() or "http error 403" in error_msg.lower():
                print(f"Mixer Engine: Download for {video_id} likely blocked (403/Sign-in). Error: {error_msg[:200]}")
            else:
                print(f"Mixer Engine: Download Failed for {video_id} (MP3 not found or empty). Output: {error_msg[:300]}")
            return None, error_msg

    except subprocess.TimeoutExpired:
        print(f"Mixer Engine: yt-dlp download timed out for {video_id}")
        return None, "Download timed out"
    except Exception as e:
        print(f"Mixer Engine: Python error during download of {video_id}: {e}")
        return None, str(e)

def spleeter_separate_2stem(input_mp3_path, video_id, base_spleeter_dir):
    video_specific_spleeter_dir = os.path.join(base_spleeter_dir, video_id)
    os.makedirs(video_specific_spleeter_dir, exist_ok=True)
    vocals_out_path = os.path.join(video_specific_spleeter_dir, 'vocals.wav')
    accomp_out_path = os.path.join(video_specific_spleeter_dir, 'accompaniment.wav')

    if os.path.exists(vocals_out_path) and os.path.exists(accomp_out_path):
        print(f"Mixer Engine: Spleeter Cache hit for {video_id}")
        return vocals_out_path, accomp_out_path

    print(f"Mixer Engine: Spleetering {input_mp3_path} (for {video_id})")
    try:
        separator = Separator('spleeter:2stems-16kHz', multiprocess=False)
        audio_loader = AudioAdapter.default()
        waveform, sr = audio_loader.load(input_mp3_path, sample_rate=16000) # Match Spleeter model's SR

        prediction = separator.separate(waveform)
        sf.write(vocals_out_path, prediction['vocals'], sr)
        sf.write(accomp_out_path, prediction['accompaniment'], sr)
        print(f"Mixer Engine: Spleeter stems saved for {video_id}")
        return vocals_out_path, accomp_out_path
    except Exception as e:
        print(f"Mixer Engine: Spleeter error for {video_id} ({input_mp3_path}): {type(e).__name__} - {e}")
        return None, None

def load_audio_for_analysis(path, target_sr=16000, load_duration=None): # Changed duration to load_duration
    try:
        # If finding best 1-min segment, load more initially (e.g., 180s for find_best_quality_segment scan_limit)
        y, sr = librosa.load(path, sr=target_sr, mono=True, duration=load_duration)
        return y, sr
    except Exception as e:
        print(f"Mixer Engine: Error loading audio {path}: {e}")
        return np.array([]), target_sr

def find_bpm_and_beats(y, sr):
    if not y.any(): return 120.0, np.array([])
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm_array = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        bpm = bpm_array[0] if isinstance(bpm_array, np.ndarray) and bpm_array.size > 0 else 120.0
        beat_track_output = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, tightness=100)
        beat_frames = beat_track_output[1] if isinstance(beat_track_output, tuple) else beat_track_output
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return bpm, beat_times
    except Exception as e:
        print(f"Mixer Engine: Beat tracking error: {e}. Defaulting.")
        return 120.0, np.array([])
    
def detect_vocal_sections(vocals_y, sr, min_energy_threshold=0.01, 
                          min_pause_duration=0.5, min_section_duration=0.8, # Shorter for fine-grained detection
                          max_section_duration=30.0):
    if not vocals_y.any(): return []
    
    frame_length = 2048
    hop_length = 512
    
    vocal_energy = librosa.feature.rms(y=vocals_y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(vocal_energy)), sr=sr, hop_length=hop_length)
    
    active_sections = []
    in_section = False
    section_start_time = 0.0

    for i in range(len(vocal_energy)):
        is_energetic = vocal_energy[i] > min_energy_threshold
        current_time = times[i]
        
        if is_energetic and not in_section:
            in_section = True
            section_start_time = current_time
        elif not is_energetic and in_section:
            section_end_time = current_time
            duration = section_end_time - section_start_time
            if min_section_duration <= duration <= max_section_duration:
                active_sections.append((section_start_time, section_end_time))
            in_section = False

    if in_section:
        section_end_time = times[-1]
        duration = section_end_time - section_start_time
        if min_section_duration <= duration <= max_section_duration:
            active_sections.append((section_start_time, section_end_time))
            
    if not active_sections: return []

    merged_sections = []
    current_start, current_end = active_sections[0]
    for i in range(1, len(active_sections)):
        next_start, next_end = active_sections[i]
        if (next_start - current_end) < min_pause_duration:
            current_end = next_end
        else:
            merged_sections.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_sections.append((current_start, current_end))
    return merged_sections

def score_segment(y_segment_audio, sr_segment_audio, 
                  y_vocals_for_segment, sr_vocals_for_segment, 
                  segment_start_time_in_song, segment_end_time_in_song, 
                  beat_times_full_song):
    score = 100.0
    segment_duration = librosa.get_duration(y=y_segment_audio, sr=sr_segment_audio)

    # Penalty for very short segments (if somehow one is passed)
    if segment_duration < 50.0 : score -= 50 # Heavily penalize if much shorter than 1 min
    if segment_duration < 30.0 : score -= 100


    # Vocal activity checks (ensure y_vocals_for_segment corresponds to y_segment_audio)
    if y_vocals_for_segment.any() and len(y_vocals_for_segment) == len(y_segment_audio): # Basic sanity check
        # Check vocals at the very start (e.g. first 0.5s)
        start_check_samples_voc = int(0.5 * sr_vocals_for_segment)
        if np.any(y_vocals_for_segment[:start_check_samples_voc] > 0.02): # Slightly higher threshold
            score -= 30  # Start mid-vocal is bad
            # print(f"Debug Score: Penalty for vocal at precise start of segment (time {segment_start_time_in_song:.2f}s)")

        # Check vocals at the very end (e.g. last 0.5s)
        end_check_samples_voc = int(0.5 * sr_vocals_for_segment)
        if np.any(y_vocals_for_segment[-end_check_samples_voc:] > 0.02):
            score -= 30  # End mid-vocal is bad
            # print(f"Debug Score: Penalty for vocal at precise end of segment (time {segment_end_time_in_song:.2f}s)")
        
        # Check for abrupt cuts of vocal phrases using detect_vocal_sections on the segment's vocals
        vocal_phrases_in_segment = detect_vocal_sections(y_vocals_for_segment, sr_vocals_for_segment)
        for phrase_start_rel, phrase_end_rel in vocal_phrases_in_segment:
            # Penalize if a phrase starts very close to segment start (cut off intro of phrase)
            if 0.0 <= phrase_start_rel < 0.3: # Phrase starts in first 0.3s of segment
                score -= 15
                # print(f"Debug Score: Penalty, phrase starts near seg start ({phrase_start_rel:.2f}s within segment)")
            # Penalize if a phrase ends very close to segment end (cut off end of phrase)
            if 0.0 <= (segment_duration - phrase_end_rel) < 0.3: # Phrase ends in last 0.3s of segment
                score -= 15
                # print(f"Debug Score: Penalty, phrase ends near seg end ({phrase_end_rel:.2f}s of {segment_duration:.2f}s)")
    
    # Bonus for starting on a beat (relative to the full song's beats)
    # Check if segment_start_time_in_song is close to any beat_time_full_song
    if beat_times_full_song.any():
        min_dist_to_beat = np.min(np.abs(beat_times_full_song - segment_start_time_in_song))
        if min_dist_to_beat < 0.1: # Starts within 0.1s of a beat
            score += 10
            # print(f"Debug Score: Bonus for starting on/near a beat (dist {min_dist_to_beat:.2f}s)")
        elif min_dist_to_beat > 0.3: # Starts significantly off-beat
            score -=5


    return score


def find_best_quality_segment(y_full, sr_full, y_vocals_full, sr_vocals_full, beat_times_full,
                              target_duration_sec=60.0,
                              scan_limit_sec=120.0, # Scan up to the first 2 minutes for starts
                              step_sec_if_no_beats=2.0): # Check segment starts every 2s if no beats
    song_total_duration = librosa.get_duration(y=y_full, sr=sr_full)

    if song_total_duration < target_duration_sec * 0.8: # If song is much shorter than target
        print(f"Mixer Engine: Song too short ({song_total_duration:.1f}s) for find_best_quality_segment. Returning what's available.")
        end_sample = min(len(y_full), int(song_total_duration * sr_full)) # Should be len(y_full)
        return y_full[:end_sample], sr_full, librosa.get_duration(y=y_full[:end_sample], sr=sr_full)

    best_segment_audio = None
    best_segment_score = -float('inf')
    best_segment_actual_duration = 0
    best_segment_start_time = 0.0

    # Define candidate start times for segments
    candidate_start_times_sec = [0.0] # Always consider segment from absolute beginning
    
    # Use beat times as primary candidates for segment starts
    if beat_times_full.any():
        for bt in beat_times_full:
            # Only consider starts if a full target_duration segment can be formed
            # and the start is within the initial scan_limit_sec of the song
            if bt < scan_limit_sec and (bt + target_duration_sec) <= song_total_duration:
                candidate_start_times_sec.append(bt)
    else: # No beats found, use fixed steps
        for start_step in np.arange(step_sec_if_no_beats, scan_limit_sec, step_sec_if_no_beats):
            if (start_step + target_duration_sec) <= song_total_duration:
                candidate_start_times_sec.append(start_step)
    
    candidate_start_times_sec = sorted(list(set(candidate_start_times_sec))) # Unique, sorted

    if not candidate_start_times_sec: # Fallback if no valid start times (e.g. song is short)
        print(f"Mixer Engine: No candidate start times found within scan limit. Using initial segment.")
        actual_end_sec = min(target_duration_sec, song_total_duration)
        end_sample = int(actual_end_sec * sr_full)
        y_segment = y_full[:end_sample]
        return y_segment, sr_full, librosa.get_duration(y=y_segment, sr=sr_full)

    print(f"Mixer Engine: Scanning {len(candidate_start_times_sec)} candidate segments (up to {scan_limit_sec:.0f}s mark)...")

    for start_sec_candidate in candidate_start_times_sec:
        start_sample = int(start_sec_candidate * sr_full)
        # Aim for target_duration_sec, but don't exceed song length
        end_sample_ideal = int((start_sec_candidate + target_duration_sec) * sr_full)
        end_sample_actual = min(end_sample_ideal, len(y_full))

        if start_sample >= end_sample_actual: continue # Segment would be empty or invalid

        y_current_segment = y_full[start_sample:end_sample_actual]
        
        y_vocals_current_segment = np.array([])
        if y_vocals_full.any():
            if sr_vocals_full == sr_full: # Expected case
                y_vocals_current_segment = y_vocals_full[start_sample:end_sample_actual]
            else: # Should not happen if preprocessing is correct
                print(f"Warning: Vocal SR mismatch during segment scoring. Vocals SR: {sr_vocals_full}, Main SR: {sr_full}")
                # Cannot reliably score vocals if SRs don't match for slicing
        
        segment_actual_end_time_in_song = start_sec_candidate + librosa.get_duration(y=y_current_segment, sr=sr_full)

        current_score = score_segment(y_current_segment, sr_full, 
                                      y_vocals_current_segment, sr_full, # Pass sr_full assuming vocals match
                                      start_sec_candidate, segment_actual_end_time_in_song,
                                      beat_times_full)
        
        # print(f"  Segment from {start_sec_candidate:.2f}s (len {librosa.get_duration(y=y_current_segment, sr=sr_full):.2f}s) - Score: {current_score:.1f}")

        if current_score > best_segment_score:
            best_segment_score = current_score
            best_segment_audio = y_current_segment
            best_segment_actual_duration = librosa.get_duration(y=best_segment_audio, sr=sr_full)
            best_segment_start_time = start_sec_candidate # Keep track of where this best segment started
            # print(f"    New Best! Start: {best_segment_start_time:.2f}s, Dur: {best_segment_actual_duration:.2f}s, Score: {best_segment_score:.1f}")


    if best_segment_audio is not None and best_segment_actual_duration > target_duration_sec * 0.75: # Ensure segment is reasonably long
        print(f"Mixer Engine: Best quality segment found (started at {best_segment_start_time:.2f}s in original song) "
              f"with score {best_segment_score:.1f}, actual duration {best_segment_actual_duration:.1f}s.")
        return best_segment_audio, sr_full, best_segment_actual_duration
    else:
        # Fallback if no suitable segment found
        print(f"Mixer Engine: No suitable best segment found (highest score {best_segment_score}). Taking first minute as fallback.")
        actual_end_sec_fallback = min(target_duration_sec, song_total_duration)
        end_sample_fallback = int(actual_end_sec_fallback * sr_full)
        y_fallback_segment = y_full[:end_sample_fallback]
        fallback_duration = librosa.get_duration(y=y_fallback_segment, sr=sr_full)
        return y_fallback_segment, sr_full, fallback_duration
    
def apply_tempo_stretch(y, sr, target_bpm, current_bpm, 
                        stretch_start_sec, stretch_end_sec, max_stretch_ratio=0.08):
    if not y.any() or current_bpm <= 0 or target_bpm <= 0 or abs(target_bpm - current_bpm) < 2: # Check if BPMs are too close or invalid
        # print(f"Mixer Engine: Skipping tempo stretch for BPM {current_bpm:.1f} -> {target_bpm:.1f} (too close or invalid)")
        return y 

    stretch_ratio = target_bpm / current_bpm
    stretch_ratio = np.clip(stretch_ratio, 1.0 - max_stretch_ratio, 1.0 + max_stretch_ratio)
    
    if abs(1.0 - stretch_ratio) < 0.01: # If change is negligible (less than 1%)
        # print(f"Mixer Engine: Tempo stretch ratio {stretch_ratio:.3f} too small, skipping.")
        return y

    start_sample = int(stretch_start_sec * sr)
    end_sample = int(stretch_end_sec * sr)
    
    # Boundary checks for samples
    if start_sample < 0: start_sample = 0
    if end_sample > len(y): end_sample = len(y)
    if start_sample >= end_sample : 
        # print(f"Mixer Engine: Invalid segment for tempo stretch (start {start_sample} >= end {end_sample})")
        return y
    
    segment_to_stretch = y[start_sample:end_sample]
    if not segment_to_stretch.any(): # Empty segment
        # print("Mixer Engine: Segment to stretch is empty.")
        return y
        
    # Ensure float32 for pyrubberband
    if segment_to_stretch.dtype != np.float32:
        segment_to_stretch = segment_to_stretch.astype(np.float32)

    try:
        # print(f"Mixer Engine: Stretching segment from {stretch_start_sec:.2f}s to {stretch_end_sec:.2f}s "
        #       f"with ratio {stretch_ratio:.3f} (BPM {current_bpm:.1f} -> {target_bpm:.1f})")
        stretched_segment = pyrb.time_stretch(segment_to_stretch, sr, stretch_ratio)
        
        # Ensure stretched_segment is 1D if input was 1D
        if y.ndim == 1 and stretched_segment.ndim > 1:
            stretched_segment = stretched_segment.squeeze() # Or take mean if stereo output not desired
        elif y.ndim > 1 and stretched_segment.ndim == 1 and y.shape[1] == stretched_segment.shape[0] and y.shape[0] != stretched_segment.shape[0]:
             # This can happen if pyrb outputs a transposed array for multichannel
             print("Warning: PyRubberband output shape mismatch, attempting to fix for multichannel.")
             # This case needs careful handling depending on pyrb's multichannel output behavior.
             # For mono input, this shouldn't be an issue.
             pass


        # Reconstruct the audio
        # Ensure all parts are 1D for concatenation if original y was 1D
        part1 = y[:start_sample]
        part3 = y[end_sample:]
        
        if y.ndim == 1:
            if part1.ndim > 1: part1 = part1.squeeze()
            if part3.ndim > 1: part3 = part3.squeeze()
            if stretched_segment.ndim > 1: stretched_segment = stretched_segment.squeeze()
            
            # Final check for empty parts before concatenation
            if not part1.any() and not stretched_segment.any() and not part3.any():
                return y # All parts became empty, return original
            
            concat_list = []
            if part1.any(): concat_list.append(part1)
            if stretched_segment.any(): concat_list.append(stretched_segment)
            if part3.any(): concat_list.append(part3)
            
            if not concat_list: return y # Should not happen if y was not empty

            return np.concatenate(concat_list)
        else: # Handle multichannel (though current code aims for mono)
             # This part would need more careful shape management for multichannel
            return np.concatenate((y[:start_sample], stretched_segment, y[end_sample:]), axis=0) # Assuming axis=0 for samples

    except Exception as e:
        print(f"Mixer Engine: Error during pyrubberband time_stretch: {type(e).__name__} - {e}")
        return y # Return original on error

# --- The rest of mixer_engine.py (determine_fixed_transition_point_sec, apply_tempo_stretch, np_audio_to_pydub, create_dj_mix, search_youtube_yt_dlp, __main__ test block)
#     can remain largely the same as in the "Complete Code" version from before.
#     The key change is that `create_dj_mix` will now receive audio segments (`y`) that are already the desired 1-minute chunks.
#     The `tp` value in `song_data_list` will be the actual duration of that 1-minute chunk.

# Make sure np_audio_to_pydub can handle potentially shorter segments correctly
def np_audio_to_pydub(y_np, sr_np):
    if not y_np.any() or len(y_np) < (sr_np * 0.1): # If less than 0.1s, treat as silent
        # print("Mixer Engine: np_audio_to_pydub received very short/empty array, returning silent.")
        return AudioSegment.silent(duration=100)
    
    if np.issubdtype(y_np.dtype, np.floating):
        peak = np.max(np.abs(y_np))
        y_normalized = y_np / peak if peak > 0 else y_np
        y_int16 = (y_normalized * 32767).astype(np.int16)
    elif y_np.dtype == np.int16:
        y_int16 = y_np
    else:
        y_int16 = y_np.astype(np.int16)

    if y_int16.ndim > 1: y_int16 = y_int16.mean(axis=1).astype(np.int16)

    try:
        return AudioSegment(y_int16.tobytes(), frame_rate=sr_np, sample_width=y_int16.dtype.itemsize, channels=1)
    except Exception as e:
        print(f"Mixer Engine: Error converting numpy to pydub: {e}. SR: {sr_np}, Shape: {y_np.shape}, Dtype: {y_np.dtype}")
        return AudioSegment.silent(duration=100)

def create_dj_mix(song_data_list, output_mix_path,
                  # segment_duration_sec is now implicit in the data
                  crossfade_duration_ms=3500, # Using value from app.py
                  tempo_match_window_sec=8.0):
    if not song_data_list: return None
    num_songs = len(song_data_list)
    
    MIX_SR = 16000 # Should match the SR of audio in song_data_list['y']

    processed_audio_segments_for_pydub = []

    for i in range(num_songs):
        current_song_proc = song_data_list[i]
        y_segment_audio = current_song_proc['y'] # This is already the ~1-min segment
        sr_segment = current_song_proc['sr']
        bpm_segment = current_song_proc['bpm']
        # 'tp' is the actual duration of this segment
        actual_segment_duration_sec = current_song_proc['tp'] 
        
        if sr_segment != MIX_SR: # Should not happen if processed correctly
            print(f"Mixer Engine WARNING: Segment SR {sr_segment} != MIX_SR {MIX_SR}. Resampling.")
            y_segment_audio = librosa.resample(y_segment_audio, orig_sr=sr_segment, target_sr=MIX_SR)
            sr_segment = MIX_SR

        # Apply tempo stretch to the *end* of this segment if it's not the last song
        if i < num_songs - 1:
            next_song_proc = song_data_list[i+1]
            bpm_next_segment = next_song_proc['bpm']
            
            # Tempo stretch window is relative to the end of the current segment
            stretch_start_sec = max(0, actual_segment_duration_sec - tempo_match_window_sec)
            stretch_end_sec = actual_segment_duration_sec # Stretch up to the very end
            
            # print(f"Mixer Engine: Tempo matching segment {i+1} (end BPM {bpm_segment:.1f}) to next segment's start (BPM {bpm_next_segment:.1f})")
            y_segment_audio = apply_tempo_stretch(
                y_segment_audio, sr_segment, bpm_next_segment, bpm_segment,
                stretch_start_sec, stretch_end_sec
            )
        
        pydub_segment = np_audio_to_pydub(y_segment_audio, sr_segment)
        
        # The segment duration is already determined by 'tp' and the extracted y_segment_audio
        # No further slicing by duration needed here unless pydub_segment is unexpectedly long
        if len(pydub_segment) > (actual_segment_duration_sec + 1) * 1000 : # If pydub segment much longer, trim
            pydub_segment = pydub_segment[:int(actual_segment_duration_sec * 1000)]

        processed_audio_segments_for_pydub.append(pydub_segment)

    if not processed_audio_segments_for_pydub: return None
        
    final_mix_segment = processed_audio_segments_for_pydub[0]
    # print(f"Mixer Engine: Starting mix with segment 1 (len: {len(final_mix_segment)/1000.0:.2f}s)")

    for i in range(1, len(processed_audio_segments_for_pydub)):
        segment_to_add = processed_audio_segments_for_pydub[i]
        # print(f"Mixer Engine: Adding segment {i+1} (len: {len(segment_to_add)/1000.0:.2f}s)")
        
        effective_crossfade = min(crossfade_duration_ms, len(final_mix_segment), len(segment_to_add))
        if effective_crossfade < 500: # Min sensible crossfade
             final_mix_segment = final_mix_segment + segment_to_add
        else:
             final_mix_segment = final_mix_segment.append(segment_to_add, crossfade=effective_crossfade)
        # print(f"Mixer Engine: Current mix total duration: {len(final_mix_segment)/1000.0:.2f}s")

    if final_mix_segment and len(final_mix_segment) > 1000:
        try:
            final_mix_segment = final_mix_segment.normalize()
            final_mix_segment.export(output_mix_path, format="mp3", bitrate="128k")
            print(f"Mixer Engine: DJ Mix created: {output_mix_path} (Duration: {len(final_mix_segment)/1000.0:.2f}s)")
            return output_mix_path
        except Exception as e:
            print(f"Mixer Engine: Error exporting final mix: {e}")
            return None
    else:
        print("Mixer Engine: Final mix segment too short or empty. Mix failed.")
        return None
    
def load_stem_waveform(path, target_sr=None): # Added target_sr for consistency
    """Load a single stem waveform."""
    try:
        # If target_sr is None, librosa loads at native SR.
        # If Spleeter always outputs at 16kHz (due to 2stems-16kHz model),
        # and your target_sr for analysis is also 16kHz, this is fine.
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        print(f"Mixer Engine: Error loading stem waveform {path}: {e}")
        return np.array([]), target_sr if target_sr else 0

# --- yt-dlp based search and __main__ test block can remain as previously provided ---
def search_youtube_yt_dlp(query_text, num_results=5):
    search_query = f"ytsearch{num_results}:{query_text}"
    results = []
    try:
        proc = subprocess.run(
            ['yt-dlp', search_query, '--dump-json', '--no-playlist', '--flat-playlist'],
            capture_output=True, text=True, check=True, timeout=15
        )
        for line in proc.stdout.strip().split('\n'):
            if not line: continue
            try:
                video_info = json.loads(line)
                if video_info.get("id") and video_info.get("title"):
                    results.append({
                        "video_id": video_info["id"],
                        "title": video_info["title"],
                        "thumbnail_url": video_info.get("thumbnail", "") 
                    })
            except json.JSONDecodeError:
                print(f"Mixer Engine: yt-dlp search - Could not parse JSON line: {line[:100]}")
        return results
    except subprocess.CalledProcessError as e:
        print(f"Mixer Engine: yt-dlp search command failed: {e.stderr if e.stderr else e.stdout}")
    except subprocess.TimeoutExpired:
        print(f"Mixer Engine: yt-dlp search timed out for query: {query_text}")
    except Exception as e:
        print(f"Mixer Engine: Unexpected error in yt-dlp search: {e}")
    return []

if __name__ == '__main__':
    # Simplified test for Option B
    print("Mixer Engine - Self-Test Mode (Option B Style)")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    YOUTUBE_API_KEY_FROM_ENV = os.getenv('YOUTUBE_API_KEY')
    if YOUTUBE_API_KEY_FROM_ENV: init_youtube_api(YOUTUBE_API_KEY_FROM_ENV)

    test_vid_id = "dQw4w9WgXcQ" # Rick Astley - short for testing
    dl_dir = "test_mixer_downloads_optionb"
    spleeter_dir_test = "test_mixer_spleeter_optionb"
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(spleeter_dir_test, exist_ok=True)
    
    mp3_file, title = download_youtube_as_mp3(test_vid_id, dl_dir)
    if mp3_file:
        print(f"Downloaded '{title}' to {mp3_file}")
        
        vocals_p, _ = spleeter_separate_2stem(mp3_file, test_vid_id, spleeter_dir_test)
        
        target_sr_test = 16000
        # Load more audio initially for find_best_quality_segment to scan
        y_full, sr_full = load_audio_for_analysis(mp3_file, target_sr=target_sr_test, load_duration=180) # Load up to 3 mins
        
        y_vocals, sr_vocals = np.array([]), target_sr_test
        if vocals_p and os.path.exists(vocals_p):
            y_v_loaded, sr_v_loaded = load_stem_waveform(vocals_p)
            if sr_v_loaded != target_sr_test: y_vocals = librosa.resample(y_v_loaded, orig_sr=sr_v_loaded, target_sr=target_sr_test)
            else: y_vocals = y_v_loaded
            sr_vocals = target_sr_test

        _, beats = find_bpm_and_beats(y_full, sr_full)
        
        y_seg, sr_seg, dur_seg = find_best_quality_segment(y_full, sr_full, y_vocals, sr_vocals, beats, target_duration_sec=30.0) # Test with 30s segments
        
        if y_seg.any():
            print(f"Best segment found: duration {dur_seg:.2f}s")
            bpm_seg, _ = find_bpm_and_beats(y_seg, sr_seg)
            print(f"Segment BPM: {bpm_seg:.1f}")
            
            # Create a dummy song_data_list for create_dj_mix test
            song_data_for_mix_test = [{
                'y': y_seg, 'sr': sr_seg, 'bpm': bpm_seg, 'tp': dur_seg, 'title': title, 'id': test_vid_id
            }]
            # You'd typically have multiple such entries for a real mix
            # For a simple test, let's just "mix" this one segment (won't do much mixing)
            if len(song_data_for_mix_test) >= 1:
                 test_mix_output = "test_final_mix_optionb.mp3"
                 result_path = create_dj_mix(song_data_for_mix_test, test_mix_output)
                 if result_path: print(f"Test mix (single segment) created: {result_path}")

        else:
            print("Failed to find best segment.")
    else:
        print(f"Download test failed for {test_vid_id}.")
    print("\nMixer Engine Self-Test Complete.")