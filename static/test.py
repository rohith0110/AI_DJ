# app.py
import os
import time
import threading
import uuid
from collections import Counter, defaultdict, deque
from dotenv import load_dotenv
import shutil
import random

from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import mixer_engine # Ensure this is the updated mixer_engine.py

# --- CONFIGURATION ---
load_dotenv()
APP_SECRET_KEY = os.urandom(24)
YOUTUBE_API_KEY_FROM_ENV = os.getenv('YOUTUBE_API_KEY')

# Directories
DOWNLOAD_DIR_BASE = 'youtube_downloads'
SPLEETER_TEMP_DIR_BASE = 'spleeter_temp'
GENERATED_MIXES_DIR = 'generated_mixes'

# Mix Settings
NUM_SONGS_FOR_MIX = 7
SEGMENT_DURATION_SEC = 60.0  # Each song segment will be 1 minute
MIX_PLAY_DURATION_APPROX = NUM_SONGS_FOR_MIX * SEGMENT_DURATION_SEC # Approx 7 mins
MIX_GENERATION_INTERVAL_SECONDS = int(MIX_PLAY_DURATION_APPROX * 0.8) # Start next gen a bit before current ends
                                                                   # Adjust this based on actual processing time
CROSSFADE_MS = 4000 # Crossfade duration between segments

# Cache and History
SONG_FILE_CACHE_EXPIRY_SECONDS = 3 * 60 * 60 # 3 hours for downloaded files
MAX_GENERATED_MIXES_TO_KEEP = 3
PLAYED_VIDEO_ID_HISTORY_SIZE = 50 # Remember last 50 songs to avoid quick repeats

# Rate limiting for voting
MAX_VOTES_PER_USER_WINDOW = 3 # Max 3 votes from one IP...
USER_VOTE_WINDOW_MINS = 15    # ...per 15 minutes

# --- FLASK APP & EXTENSIONS ---
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["300 per day", "100 per hour"],
    storage_uri="memory://",
)

# --- GLOBAL STATE & LOCKS ---
votes = Counter() # {video_id: count}
video_id_details_cache = {} # {video_id: {'title': ..., 'thumbnail_url': ..., 'youtube_url': ...}}
# song_file_cache: {video_id: {'mp3_path': ..., 'stems_path': {'v':..., 'a':...}, 'last_used': ts}} NO LONGER USED, mixer_engine handles its own file cache
played_video_ids = deque(maxlen=PLAYED_VIDEO_ID_HISTORY_SIZE) # FIFO queue of recently played video_ids

current_mix_info = { # Renamed from current_mix_details for clarity
    "url": None,
    "songs_in_mix_titles": [], # List of titles of songs in the current mix
    "songs_in_mix_ids": [], # List of video_ids in the current mix
    "generated_at_ts": 0,
    "next_mix_generation_starts_at_ts": 0,
    "estimated_processing": False # Flag if current processing takes long
}

user_vote_timestamps = defaultdict(list)

# Locks
votes_lock = threading.Lock()
details_cache_lock = threading.Lock()
mix_info_lock = threading.Lock()
played_ids_lock = threading.Lock()

# --- INITIALIZATION ---
def initialize_system():
    if not YOUTUBE_API_KEY_FROM_ENV:
        print("FATAL: YOUTUBE_API_KEY not found. Set it in .env")
        exit(1)
    mixer_engine.init_youtube_api(YOUTUBE_API_KEY_FROM_ENV) # Pass key to mixer_engine
    
    for d in [DOWNLOAD_DIR_BASE, SPLEETER_TEMP_DIR_BASE, GENERATED_MIXES_DIR]:
        os.makedirs(d, exist_ok=True)
    
    print(f"System initialized. Target mix duration: ~{MIX_PLAY_DURATION_APPROX // 60} mins.")
    print(f"Mix generation will start ~{MIX_GENERATION_INTERVAL_SECONDS // 60} mins into current mix playback.")
    update_next_mix_gen_time_estimate()

def update_next_mix_gen_time_estimate(processing_just_finished=False):
    with mix_info_lock:
        if processing_just_finished or current_mix_info["generated_at_ts"] == 0:
            # If just finished, or first run, schedule next based on interval from now
            current_mix_info["next_mix_generation_starts_at_ts"] = time.time() + MIX_GENERATION_INTERVAL_SECONDS
        else:
            # Schedule relative to when current mix started playing + interval
            current_mix_info["next_mix_generation_starts_at_ts"] = \
                current_mix_info["generated_at_ts"] + MIX_GENERATION_INTERVAL_SECONDS
        current_mix_info["estimated_processing"] = False # Reset processing flag


# --- YOUTUBE SEARCH & VOTE ---
@app.route('/search_youtube', methods=['POST'])
@limiter.limit("15 per minute")
def search_youtube_route():
    query = request.json.get('query', '').strip()
    if not query: return jsonify({"error": "Query empty."}), 400

    try:
        # mixer_engine.youtube should be initialized
        search_req = mixer_engine.youtube.search().list(
            part='snippet', q=query, type='video', maxResults=5, videoDefinition="high" # prefer HD
        ).execute()
        
        results = []
        if search_req.get('items'):
            for item in search_req['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                thumbnail = item['snippet']['thumbnails']['default']['url']
                results.append({"video_id": video_id, "title": title, "thumbnail_url": thumbnail})
                # Cache details immediately
                with details_cache_lock:
                    if video_id not in video_id_details_cache:
                         video_id_details_cache[video_id] = {"title": title, "thumbnail_url": thumbnail, "youtube_url": f"https://www.youtube.com/watch?v={video_id}"}
        return jsonify(results)
    except Exception as e:
        print(f"YouTube search error: {e}")
        return jsonify({"error": "YouTube search failed."}), 500

def check_user_vote_limit(ip_address):
    now = time.time()
    cooldown_seconds = USER_VOTE_WINDOW_MINS * 60
    user_vote_timestamps[ip_address] = [ts for ts in user_vote_timestamps[ip_address] if now - ts < cooldown_seconds]
    if len(user_vote_timestamps[ip_address]) < MAX_VOTES_PER_USER_WINDOW:
        user_vote_timestamps[ip_address].append(now)
        return True, f"Vote accepted. {MAX_VOTES_PER_USER_WINDOW - len(user_vote_timestamps[ip_address])} votes left in {USER_VOTE_WINDOW_MINS}m."
    else:
        return False, f"Rate limit: Max {MAX_VOTES_PER_USER_WINDOW} votes per {USER_VOTE_WINDOW_MINS}m."

@app.route('/cast_vote', methods=['POST'])
@limiter.limit("20 per minute") # Per IP limit
def cast_vote_route():
    video_id = request.json.get('video_id', '').strip()
    if not video_id: return jsonify({"error": "Video ID missing."}), 400

    ip = get_remote_address()
    allowed, msg = check_user_vote_limit(ip)
    if not allowed: return jsonify({"success": False, "message": msg, "limit_hit": True}), 429

    with votes_lock: votes[video_id] += 1
    
    # Ensure details are cached if vote comes from elsewhere (e.g. direct link)
    with details_cache_lock:
        if video_id not in video_id_details_cache:
            details = mixer_engine.get_youtube_video_details(video_id) # Fetches from YouTube
            if details: video_id_details_cache[video_id] = {"title": details['title'], "thumbnail_url": "", "youtube_url": f"https://www.youtube.com/watch?v={video_id}"}
            else: video_id_details_cache[video_id] = {"title": "Title N/A", "thumbnail_url": "", "youtube_url": f"https://www.youtube.com/watch?v={video_id}"}


    print(f"Vote for {video_id} (now {votes[video_id]}) from {ip}. Msg: {msg}")
    return jsonify({"success": True, "message": msg})

# --- MIX GENERATION THREAD ---
def song_processing_worker(video_id, target_sr=16000):
    """Worker to download, spleeter (if needed), and analyze one song."""
    try:
        mp3_path, title = mixer_engine.download_youtube_as_mp3(video_id, DOWNLOAD_DIR_BASE)
        if not mp3_path:
            return video_id, None, f"Download failed for {video_id}"

        # Spleeter - this is the slowest part. mixer_engine handles caching.
        vocals_p, _ = mixer_engine.spleeter_separate_2stem(mp3_path, video_id, SPLEETER_TEMP_DIR_BASE)
        # Note: We don't strictly need stems for fixed duration, but if tempo matching is used, it's good.
        # For fixed 1-min segments, vocal analysis for transition points is less critical.

        y_audio, sr_audio = mixer_engine.load_audio_for_analysis(mp3_path, target_sr=target_sr)
        bpm, beats = mixer_engine.find_bpm_and_beats(y_audio, sr_audio)
        
        # Transition point is fixed (e.g., 60s), but ensure it's valid for the song's length
        tp_sec = mixer_engine.determine_fixed_transition_point_sec(y_audio, sr_audio, fixed_duration_sec=SEGMENT_DURATION_SEC)

        return video_id, {
            'y': y_audio, 'sr': sr_audio, 'bpm': bpm, 'tp': tp_sec, 
            'path': mp3_path, 'title': title, 'id': video_id
        }, None
    except Exception as e:
        print(f"Error in song_processing_worker for {video_id}: {e}")
        return video_id, None, str(e)


def mix_generation_manager():
    print("Mix Generation Manager Thread Started.")
    time.sleep(5) # Initial startup grace period

    while True:
        # When should we start generating the next mix?
        with mix_info_lock:
            time_to_next_gen_start = current_mix_info["next_mix_generation_starts_at_ts"] - time.time()
            is_processing_now = current_mix_info["estimated_processing"]

        if time_to_next_gen_start > 0 and not is_processing_now:
            # print(f"MixGen: Sleeping for {time_to_next_gen_start:.0f}s until next schedule.")
            time.sleep(min(time_to_next_gen_start, 60)) # Sleep adaptively, check every minute max
            continue
        
        if is_processing_now: # Already processing, wait
            time.sleep(30)
            continue

        # --- Time to generate a new mix ---
        print(f"[{time.strftime('%H:%M:%S')}] MixGen: Cycle Start. Selecting songs.")
        with mix_info_lock: current_mix_info["estimated_processing"] = True # Mark as processing

        # Select top N songs that haven't been played recently
        selected_video_ids_for_mix = []
        with votes_lock, played_ids_lock, details_cache_lock:
            # Get more than N to filter out played ones
            candidate_votes = votes.most_common(NUM_SONGS_FOR_MIX * 3 + len(played_video_ids))
            for vid, count in candidate_votes:
                if count == 0: break # No more votes
                if vid not in played_video_ids:
                    selected_video_ids_for_mix.append(vid)
                    if len(selected_video_ids_for_mix) >= NUM_SONGS_FOR_MIX:
                        break
            
            # If not enough unique unplayed songs, pick random unplayed from available votes, or even from history if desperate
            if len(selected_video_ids_for_mix) < NUM_SONGS_FOR_MIX:
                print(f"MixGen: Only {len(selected_video_ids_for_mix)} unplayed top songs. Trying to fill with other voted songs.")
                all_voted_ids = list(votes.keys())
                random.shuffle(all_voted_ids) # Shuffle to get variety
                for vid in all_voted_ids:
                    if vid not in selected_video_ids_for_mix and vid not in played_video_ids:
                        selected_video_ids_for_mix.append(vid)
                        if len(selected_video_ids_for_mix) >= NUM_SONGS_FOR_MIX:
                            break
            
            # If still not enough, and history is allowed to be re-used (after a while)
            # This part can be complex logic for fallback song selection

        if len(selected_video_ids_for_mix) < NUM_SONGS_FOR_MIX:
            print(f"MixGen: Insufficient unique/unplayed songs ({len(selected_video_ids_for_mix)}). Need {NUM_SONGS_FOR_MIX}. Waiting.")
            with mix_info_lock: current_mix_info["estimated_processing"] = False
            update_next_mix_gen_time_estimate(processing_just_finished=True) # Reschedule sooner
            time.sleep(60) # Wait a bit before retrying selection
            continue
        
        print(f"MixGen: Selected IDs for processing: {selected_video_ids_for_mix}")

        # Process songs in parallel (download, spleeter, analyze)
        # Target SR for audio loaded into memory, e.g., Spleeter's 16kHz
        TARGET_PROCESS_SR = 16000
        processed_song_data_map = {} # video_id -> data_dict
        
        # Use ThreadPoolExecutor for parallel song processing
        # Limit workers to avoid overwhelming CPU, especially with Spleeter
        # Spleeter itself might not be very thread-safe or efficient with many parallel runs on one GPU/CPU
        # For CPU, 2-3 workers might be a good balance. For GPU, often 1 worker for Spleeter is best.
        # Let's assume CPU and moderate parallelism for downloads/analysis.
        max_workers_song_proc = min(NUM_SONGS_FOR_MIX, 3) # Adjust as needed
        
        processing_start_time = time.time()
        print(f"MixGen: Starting parallel processing of {len(selected_video_ids_for_mix)} songs with {max_workers_song_proc} workers...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_song_proc) as executor:
            future_to_vid = {
                executor.submit(song_processing_worker, vid, TARGET_PROCESS_SR): vid 
                for vid in selected_video_ids_for_mix
            }
            for future in concurrent.futures.as_completed(future_to_vid):
                vid = future_to_vid[future]
                try:
                    original_vid, data, error_msg = future.result()
                    if error_msg:
                        print(f"MixGen: Error processing song {original_vid}: {error_msg}")
                    elif data:
                        processed_song_data_map[original_vid] = data
                        print(f"MixGen: Successfully processed {original_vid} - {data['title']}")
                except Exception as exc:
                    print(f"MixGen: Song {vid} generated an exception during future.result(): {exc}")

        processing_duration = time.time() - processing_start_time
        print(f"MixGen: Parallel song processing finished in {processing_duration:.2f} seconds.")

        # Order the successfully processed songs according to original selection order (important for mix flow)
        final_song_data_for_mix = []
        mix_song_titles = []
        mix_song_ids_final = []

        for vid in selected_video_ids_for_mix:
            if vid in processed_song_data_map:
                data = processed_song_data_map[vid]
                final_song_data_for_mix.append(data)
                mix_song_titles.append(data['title'])
                mix_song_ids_final.append(vid)
            if len(final_song_data_for_mix) >= NUM_SONGS_FOR_MIX: # Should match NUM_SONGS_FOR_MIX
                break
        
        if len(final_song_data_for_mix) < NUM_SONGS_FOR_MIX:
            print(f"MixGen: Not enough songs successfully processed ({len(final_song_data_for_mix)}). Need {NUM_SONGS_FOR_MIX}. Aborting mix.")
            with mix_info_lock: current_mix_info["estimated_processing"] = False
            update_next_mix_gen_time_estimate(processing_just_finished=True)
            # No sleep here, loop will check schedule
            continue

        # Create the actual DJ mix
        mix_ts = int(time.time())
        new_mix_fname = f"mix_{mix_ts}.mp3"
        new_mix_fpath = os.path.join(GENERATED_MIXES_DIR, new_mix_fname)

        print(f"MixGen: Creating final mix with {len(final_song_data_for_mix)} songs -> {new_mix_fpath}")
        
        # mixer_engine.create_dj_mix expects list of song_data dicts
        generated_mix_path = mixer_engine.create_dj_mix(
            final_song_data_for_mix,
            new_mix_fpath,
            segment_duration_sec=SEGMENT_DURATION_SEC,
            crossfade_duration_ms=CROSSFADE_MS
        )

        if generated_mix_path:
            print(f"MixGen: New mix generated: {generated_mix_path}")
            with mix_info_lock, played_ids_lock, votes_lock:
                current_mix_info["url"] = f"/{GENERATED_MIXES_DIR}/{new_mix_fname}"
                current_mix_info["songs_in_mix_titles"] = mix_song_titles
                current_mix_info["songs_in_mix_ids"] = mix_song_ids_final
                current_mix_info["generated_at_ts"] = mix_ts
                
                # Add played songs to history and clear their votes (or reduce significantly)
                for vid_played in mix_song_ids_final:
                    if vid_played not in played_video_ids: # deque handles maxlen
                        played_video_ids.append(vid_played)
                    if vid_played in votes:
                        del votes[vid_played] # Reset votes for played songs
            
            cleanup_old_mixes_on_disk()
            # No need for song_file_cache cleanup here, mixer_engine handles its own file based cache
        else:
            print(f"MixGen: Mix creation FAILED for some reason.")
        
        with mix_info_lock: current_mix_info["estimated_processing"] = False
        update_next_mix_gen_time_estimate(processing_just_finished=True)
        print(f"MixGen: Cycle complete. Next generation cycle starts around {time.ctime(current_mix_info['next_mix_generation_starts_at_ts'])}")
        # No explicit sleep here, the loop's check on `time_to_next_gen_start` will handle it.

# --- STATUS & PLAYBACK ROUTES ---
@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/status')
def status_route():
    with mix_info_lock, votes_lock, details_cache_lock, played_ids_lock:
        # Prepare top voted preview (excluding recently played)
        top_voted_preview_list = []
        # Get more candidates to filter
        candidate_top_votes = votes.most_common(15) 
        for vid, count in candidate_top_votes:
            if vid not in played_video_ids: # Show only unplayed in "top voted"
                title = video_id_details_cache.get(vid, {}).get("title", vid)
                top_voted_preview_list.append({"title": title, "votes": count, "video_id": vid})
                if len(top_voted_preview_list) >= 5: # Show top 5 unplayed
                    break
        
        # If current_mix_info has a valid generated_at_ts, next_gen_starts is relative to that.
        # Otherwise, it's relative to now + interval (handled by update_next_mix_gen_time_estimate)
        next_gen_starts_in_sec = max(0, int(current_mix_info["next_mix_generation_starts_at_ts"] - time.time()))
        
        # Make a copy of current_mix_info for thread safety if it's complex
        # But here, individual fields are read under lock.
        status_payload = {
            "current_mix_url": current_mix_info["url"],
            "current_mix_song_titles": current_mix_info["songs_in_mix_titles"],
            "current_mix_generated_at_ts": current_mix_info["generated_at_ts"],
            "top_voted_preview": top_voted_preview_list,
            "next_mix_generation_starts_in_seconds": next_gen_starts_in_sec,
            "is_server_processing_mix": current_mix_info["estimated_processing"]
        }
        return jsonify(status_payload)

@app.route(f'/{GENERATED_MIXES_DIR}/<filename>')
def serve_mix_file_route(filename):
    if ".." in filename or "/" in filename: return "Invalid filename", 400
    return send_from_directory(GENERATED_MIXES_DIR, filename, mimetype='audio/mpeg')

# --- CLEANUP ---
def cleanup_old_mixes_on_disk():
    # This function is simple, assumes it's called infrequently enough not to need its own lock for GENERATED_MIXES_DIR access
    # If called very often from multiple threads, locking for listdir/remove might be needed.
    # Called from single mix_gen_thread, so should be fine.
    mix_files = sorted(
        [os.path.join(GENERATED_MIXES_DIR, f) for f in os.listdir(GENERATED_MIXES_DIR) if f.startswith("mix_") and f.endswith(".mp3")],
        key=os.path.getmtime # Sort by modification time (oldest first)
    )
    while len(mix_files) > MAX_GENERATED_MIXES_TO_KEEP:
        oldest_mix = mix_files.pop(0)
        try:
            os.remove(oldest_mix)
            print(f"Cleanup: Removed old mix file {oldest_mix}")
        except OSError as e:
            print(f"Cleanup: Error removing {oldest_mix}: {e}")

# No specific cleanup for youtube_downloads/spleeter_temp here, as mixer_engine's caching is file-based.
# A separate cron job or periodic task could clean very old, unreferenced video_id folders if desired.

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    initialize_system()
    # Start the background mix generation thread
    mix_thread = threading.Thread(target=mix_generation_manager, daemon=True)
    mix_thread.start()
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=True can mess with threading