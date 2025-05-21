# app.py
import os
import time
import threading
from collections import Counter, defaultdict, deque
from dotenv import load_dotenv
import shutil
import random
import concurrent.futures # For parallel song processing
import numpy as np
import librosa # For audio processing

from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from cachetools import TTLCache # For API search caching

import mixer_engine

# --- CONFIGURATION ---
load_dotenv()
APP_SECRET_KEY = os.urandom(24)
YOUTUBE_API_KEY_FROM_ENV = os.getenv('YOUTUBE_API_KEY') # Loaded by mixer_engine too

# Directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOWNLOAD_DIR_BASE = os.path.join(BASE_DIR, 'youtube_downloads')
SPLEETER_TEMP_DIR_BASE = os.path.join(BASE_DIR, 'spleeter_temp')
GENERATED_MIXES_DIR = os.path.join(BASE_DIR, 'generated_mixes')

# Mix Settings
NUM_SONGS_FOR_MIX = 7
SEGMENT_DURATION_SEC = 60.0
MIX_PLAY_DURATION_APPROX = NUM_SONGS_FOR_MIX * SEGMENT_DURATION_SEC
MIX_GENERATION_INTERVAL_SECONDS = int(MIX_PLAY_DURATION_APPROX * 0.75) # Start next gen earlier
CROSSFADE_MS = 3500

# Cache and History
API_SEARCH_CACHE_TTL_SECONDS = 15 * 60 # Cache API search results for 15 mins
MAX_GENERATED_MIXES_TO_KEEP = 3
PLAYED_VIDEO_ID_HISTORY_SIZE = NUM_SONGS_FOR_MIX * 3 # Avoid repeats for last 3 mixes

# Rate limiting for voting
MAX_VOTES_PER_USER_WINDOW = 1000
USER_VOTE_WINDOW_MINS = 10

# Search method: 'API' or 'YTDLP'
# Set to 'YTDLP' to try yt-dlp based search if API quota is an issue.
# YTDLP search is generally slower and might be less reliable.
SEARCH_METHOD = 'YTDLP' # or 'YTDLP'

# --- FLASK APP & EXTENSIONS ---
app = Flask(__name__)
app.secret_key = APP_SECRET_KEY
limiter = Limiter(
    get_remote_address, app=app, default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# --- GLOBAL STATE & LOCKS ---
votes_counter = Counter()
# video_id_details_cache: {video_id: {'title': ..., 'thumbnail_url': ...}}
# This cache will store details obtained either via API or yt-dlp search/download title extraction
video_id_details_cache = TTLCache(maxsize=1000, ttl=24 * 60 * 60) # Cache details for 1 day
played_video_ids_history = deque(maxlen=PLAYED_VIDEO_ID_HISTORY_SIZE)

current_mix_state = {
    "url": None, "songs_in_mix_titles": [], "songs_in_mix_ids": [],
    "generated_at_ts": 0, "next_mix_generation_starts_at_ts": 0,
    "is_server_processing_mix": False
}
user_vote_timestamps_tracker = defaultdict(list)
api_search_response_cache = TTLCache(maxsize=200, ttl=API_SEARCH_CACHE_TTL_SECONDS)

# Locks
votes_lock = threading.Lock()
details_cache_lock = threading.Lock()
mix_state_lock = threading.Lock()
played_ids_lock = threading.Lock()
api_search_cache_lock = threading.Lock()

# --- INITIALIZATION ---
def initialize_app_components():
    print("App: Initializing components...")
    if YOUTUBE_API_KEY_FROM_ENV:
        mixer_engine.init_youtube_api(YOUTUBE_API_KEY_FROM_ENV)
    else:
        print("App WARNING: YOUTUBE_API_KEY not found in .env. API search will not work.")
        global SEARCH_METHOD
        if SEARCH_METHOD == 'API':
            print("App WARNING: Switching SEARCH_METHOD to 'YTDLP' due to missing API key.")
            SEARCH_METHOD = 'YTDLP'
            
    for d in [DOWNLOAD_DIR_BASE, SPLEETER_TEMP_DIR_BASE, GENERATED_MIXES_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"App: Ensured directory exists: {d}")
    
    print(f"App: Target mix duration: ~{MIX_PLAY_DURATION_APPROX // 60} mins.")
    print(f"App: Mix generation interval: ~{MIX_GENERATION_INTERVAL_SECONDS // 60} mins.")
    print(f"App: Search method configured: {SEARCH_METHOD}")
    update_next_mix_schedule_time(first_run=True)
    print("App: Initialization complete.")

def update_next_mix_schedule_time(processing_just_finished=False, first_run=False):
    with mix_state_lock:
        now = time.time()
        if first_run: # Initial schedule, start sooner
            current_mix_state["next_mix_generation_starts_at_ts"] = now + 30 # Start first gen quickly
        elif processing_just_finished:
            # If a mix was just made, schedule based on its length minus some overlap
            # Or simply from now + interval if current_mix_state["generated_at_ts"] is recent
            current_mix_state["next_mix_generation_starts_at_ts"] = \
                current_mix_state["generated_at_ts"] + MIX_GENERATION_INTERVAL_SECONDS
        else: # Fallback if called unexpectedly, schedule from now
            current_mix_state["next_mix_generation_starts_at_ts"] = now + MIX_GENERATION_INTERVAL_SECONDS
        
        # Ensure next generation time is not in the past
        if current_mix_state["next_mix_generation_starts_at_ts"] < now:
             current_mix_state["next_mix_generation_starts_at_ts"] = now + 60 # If past, try in 1 min

        current_mix_state["is_server_processing_mix"] = False # Reset processing flag

# --- ROUTES ---
@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/search_songs', methods=['POST'])
@limiter.limit("10 per minute") # Limit search endpoint
def search_songs_route():
    query = request.json.get('query', '').strip().lower()
    if not query: return jsonify({"error": "Search query cannot be empty."}), 400

    # Check API search cache first
    with api_search_cache_lock:
        if query in api_search_response_cache:
            print(f"App: API Search cache hit for: '{query}'")
            return jsonify(api_search_response_cache[query])

    results = []
    if SEARCH_METHOD == 'API' and mixer_engine.youtube_service:
        try:
            print(f"App: Performing API search for: '{query}'")
            search_req = mixer_engine.youtube_service.search().list(
                part='snippet', q=query, type='video', maxResults=5,
                videoCategoryId='10', # Music category
                relevanceLanguage='en' # Optional: prioritize English results
            ).execute()
            if search_req.get('items'):
                for item in search_req['items']:
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    thumbnail = item['snippet']['thumbnails']['default']['url']
                    results.append({"video_id": video_id, "title": title, "thumbnail_url": thumbnail})
                    with details_cache_lock: # Cache details found via API search
                        video_id_details_cache[video_id] = {"title": title, "thumbnail_url": thumbnail}
        except Exception as e:
            print(f"App: YouTube API search error: {e}")
            # Fallback to yt-dlp search if API fails and YTDLP is allowed as fallback
            if SEARCH_METHOD == 'API': # Only try YTDLP if primary was API and it failed
                print("App: API search failed, attempting yt-dlp search as fallback...")
                results = mixer_engine.search_youtube_yt_dlp(query, num_results=5)
    
    elif SEARCH_METHOD == 'YTDLP' or not mixer_engine.youtube_service : # Use yt-dlp if primary or API not init
        print(f"App: Performing yt-dlp search for: '{query}' (API key missing or YTDLP method set)")
        results = mixer_engine.search_youtube_yt_dlp(query, num_results=5)

    # Cache yt-dlp results too (if any)
    if results:
        with api_search_cache_lock:
            api_search_response_cache[query] = results
        with details_cache_lock: # Also cache details from yt-dlp search results
            for res_item in results:
                if res_item['video_id'] not in video_id_details_cache:
                    video_id_details_cache[res_item['video_id']] = {
                        "title": res_item['title'], 
                        "thumbnail_url": res_item.get('thumbnail_url', '')
                    }
    
    return jsonify(results)


def can_user_vote(ip_address):
    now = time.time()
    cooldown_period_seconds = USER_VOTE_WINDOW_MINS * 60
    
    # Clean up old timestamps for this IP
    user_vote_timestamps_tracker[ip_address] = [
        ts for ts in user_vote_timestamps_tracker[ip_address] if now - ts < cooldown_period_seconds
    ]
    if len(user_vote_timestamps_tracker[ip_address]) < MAX_VOTES_PER_USER_WINDOW:
        user_vote_timestamps_tracker[ip_address].append(now)
        remaining_votes = MAX_VOTES_PER_USER_WINDOW - len(user_vote_timestamps_tracker[ip_address])
        return True, f"Vote counted! You have {remaining_votes} votes left in the next {USER_VOTE_WINDOW_MINS} minutes."
    else:
        # Find when the oldest vote in the window expires
        time_to_next_vote = int(cooldown_period_seconds - (now - min(user_vote_timestamps_tracker[ip_address])))
        return False, f"Vote limit reached. Try again in about {time_to_next_vote // 60}m {time_to_next_vote % 60}s."


@app.route('/submit_vote', methods=['POST'])
@limiter.limit("15 per minute") # General limit on voting endpoint per IP
def submit_vote_route():
    video_id = request.json.get('video_id', '').strip()
    if not video_id: return jsonify({"error": "Video ID is required."}), 400

    user_ip = get_remote_address()
    is_allowed, message = can_user_vote(user_ip)
    if not is_allowed:
        return jsonify({"success": False, "message": message, "limit_hit": True}), 429

    with votes_lock: votes_counter[video_id] += 1
    
    # Ensure details are cached for this video_id if not already
    with details_cache_lock:
        if video_id not in video_id_details_cache:
            # Try to fetch details using yt-dlp's download function (which also gets title)
            # or a dedicated API call if available and preferred
            print(f"App: Details for {video_id} not in cache. Attempting to fetch during vote.")
            if mixer_engine.youtube_service: # Prefer API if available for details
                details = mixer_engine.get_youtube_video_details_via_api(video_id)
                if details:
                     video_id_details_cache[video_id] = {"title": details['title'], "thumbnail_url": details.get('thumbnail_url','')}
                else: # Fallback if API fails
                    _, title_from_dl = mixer_engine.download_youtube_as_mp3(video_id, DOWNLOAD_DIR_BASE, use_api_for_title=False) # Just get title via yt-dlp
                    video_id_details_cache[video_id] = {"title": title_from_dl, "thumbnail_url": ""} # No thumbnail from this method
            else: # No API service, rely on yt-dlp during download (which might happen later)
                 # For now, store a placeholder if yt-dlp search didn't cache it
                 video_id_details_cache[video_id] = {"title": f"Video ID: {video_id}", "thumbnail_url": ""}


    print(f"App: Vote for {video_id} from {user_ip}. Total: {votes_counter[video_id]}. Message: {message}")
    return jsonify({"success": True, "message": message})


@app.route('/current_status')
def current_status_route():
    with mix_state_lock, votes_lock, details_cache_lock, played_ids_lock:
        # Prepare top voted preview (excluding recently played)
        top_voted_display_list = []
        # Get more candidates initially to filter out played ones
        candidate_votes_list = votes_counter.most_common(NUM_SONGS_FOR_MIX * 2 + len(played_video_ids_history))
        
        for vid, count in candidate_votes_list:
            if count == 0: break 
            if vid not in played_video_ids_history:
                details = video_id_details_cache.get(vid)
                title = details['title'] if details else vid # Use vid as fallback title
                top_voted_display_list.append({"title": title, "votes": count, "video_id": vid})
                if len(top_voted_display_list) >= 7: # Show top 5 unplayed
                    break
        
        next_gen_countdown_sec = max(0, int(current_mix_state["next_mix_generation_starts_at_ts"] - time.time()))
        
        payload = {
            "current_mix_url": current_mix_state["url"],
            "current_mix_song_titles": current_mix_state["songs_in_mix_titles"],
            "current_mix_generated_at_ts": current_mix_state["generated_at_ts"],
            "top_voted_preview": top_voted_display_list,
            "next_mix_generation_starts_in_seconds": next_gen_countdown_sec,
            "is_server_processing_mix": current_mix_state["is_server_processing_mix"]
        }
        return jsonify(payload)

@app.route(f'/{os.path.basename(GENERATED_MIXES_DIR)}/<filename>')
def serve_generated_mix_route(filename):
    # Basic security check for filename
    if ".." in filename or filename.count("/") > 0:
        return "Invalid filename requested.", 400
    return send_from_directory(GENERATED_MIXES_DIR, filename, mimetype='audio/mpeg')


# --- BACKGROUND MIX GENERATION THREAD ---
def song_processing_task_worker(video_id, target_sr=16000):
    """
    Downloads (if needed), Spleeters (if needed), finds the best ~1-minute audio segment,
    and analyzes it for one song.
    Returns: (video_id, song_data_dict, error_message_str)
             song_data_dict contains {'y': segment_audio, 'sr': segment_sr, ...}
    """
    print(f"MixGenWorker: Starting processing for {video_id}")
    title_from_dl_or_cache = video_id # Fallback title

    try:
        # Step 1: Download MP3 (mixer_engine handles its own file caching)
        # Determine if API should be used for title based on global SEARCH_METHOD
        # This is a bit indirect; ideally, this worker wouldn't depend on global app config like SEARCH_METHOD
        # but for now, it's simpler.
        should_use_api_for_title = (SEARCH_METHOD == 'API') and mixer_engine.youtube_service
        
        mp3_path, title_from_dl = mixer_engine.download_youtube_as_mp3(
            video_id, DOWNLOAD_DIR_BASE, 
            use_api_for_title=should_use_api_for_title
        )
        if not mp3_path:
            return video_id, None, f"Download failed for {video_id}"
        
        title_from_dl_or_cache = title_from_dl # Update title

        # Update global details cache with the title obtained (either from API during DL or yt-dlp)
        with details_cache_lock:
            cached_detail = video_id_details_cache.get(video_id, {})
            video_id_details_cache[video_id] = {
                "title": title_from_dl, 
                "thumbnail_url": cached_detail.get("thumbnail_url", "") # Keep existing thumbnail if any
            }

        # Step 2: Spleeter (mixer_engine handles its own file caching for stems)
        print(f"MixGenWorker: Attempting Spleeter for {video_id} using {mp3_path}")
        vocals_path, _ = mixer_engine.spleeter_separate_2stem(mp3_path, video_id, SPLEETER_TEMP_DIR_BASE)
        
        # Step 3: Load FULL audio for segment finding and corresponding vocal stems
        # Load a longer portion initially to give find_best_quality_segment enough material to scan.
        # The scan_limit_sec within find_best_quality_segment will determine how much of this is used.
        initial_load_duration_for_scan = 180.0 # e.g., load up to 3 minutes for scanning
        
        print(f"MixGenWorker: Loading full audio for {video_id} (up to {initial_load_duration_for_scan}s)")
        y_full_audio, sr_full_audio = mixer_engine.load_audio_for_analysis(
            mp3_path, 
            target_sr=target_sr, 
            load_duration=initial_load_duration_for_scan
        )
        if not y_full_audio.any():
            return video_id, None, f"Failed to load full audio from {mp3_path} for segment analysis: {video_id}"

        y_vocals_full = np.array([]) # Initialize as empty
        # sr_vocals_full should match target_sr after processing
        
        if vocals_path and os.path.exists(vocals_path):
            print(f"MixGenWorker: Loading full vocal stem for {video_id} from {vocals_path}")
            try:
                # load_stem_waveform loads at native SR, then we resample if needed
                y_vocals_loaded_native, sr_vocals_loaded_native = mixer_engine.load_stem_waveform(vocals_path,target_sr=target_sr)
                
                # Ensure vocal stem is also loaded for up to initial_load_duration_for_scan
                # and resampled to target_sr if necessary.
                # Note: load_stem_waveform doesn't take a duration, so it loads the whole stem.
                # We need to ensure it corresponds to the y_full_audio's loaded duration.
                # This is tricky if Spleeter produced stems from a differently truncated audio.
                # For simplicity, assume Spleeter ran on the same audio that y_full_audio comes from.
                # We'll resample it to target_sr.
                
                if sr_vocals_loaded_native != target_sr:
                    print(f"MixGenWorker: Resampling vocals for {video_id} from {sr_vocals_loaded_native}Hz to {target_sr}Hz")
                    y_vocals_full = librosa.resample(y_vocals_loaded_native, orig_sr=sr_vocals_loaded_native, target_sr=target_sr)
                else:
                    y_vocals_full = y_vocals_loaded_native
                
                # Trim y_vocals_full to match the duration of y_full_audio if necessary
                # This ensures they correspond if y_full_audio was truncated by load_duration
                max_len = len(y_full_audio)
                if len(y_vocals_full) > max_len:
                    y_vocals_full = y_vocals_full[:max_len]
                elif len(y_vocals_full) < max_len and len(y_vocals_full) > 0: # Vocals shorter, pad (less ideal)
                    print(f"MixGenWorker WARNING: Vocal stem for {video_id} is shorter than main audio. Padding with zeros.")
                    y_vocals_full = np.pad(y_vocals_full, (0, max_len - len(y_vocals_full)))


            except Exception as e_voc_load:
                print(f"MixGenWorker: Error loading/processing full vocal stem for {video_id}: {e_voc_load}")
                y_vocals_full = np.array([]) # Ensure it's an empty array on failure
        else:
            print(f"MixGenWorker: No vocal stem found or Spleeter failed for {video_id}. Proceeding without vocal analysis for segment finding.")
            y_vocals_full = np.array([])


        # Get beat times for the loaded full audio portion
        print(f"MixGenWorker: Finding beats for full audio of {video_id}")
        _, beat_times_full = mixer_engine.find_bpm_and_beats(y_full_audio, sr_full_audio)

        # Step 4: Find the best quality ~1-minute segment from the loaded audio
        print(f"MixGenWorker: Finding best quality segment for {video_id}")
        y_best_segment, sr_segment, actual_segment_duration_sec = mixer_engine.find_best_quality_segment(
            y_full_audio, sr_full_audio,
            y_vocals_full, sr_full_audio, # Pass sr_full_audio as sr_vocals_full assumes they are matched
            beat_times_full,
            target_duration_sec=SEGMENT_DURATION_SEC # Global constant for 60s
        )

        if y_best_segment is None or not y_best_segment.any():
            return video_id, None, f"Could not find/extract a suitable segment for {video_id}"

        # Step 5: Analyze the CHOSEN SEGMENT for BPM (for subsequent tempo matching in create_dj_mix)
        print(f"MixGenWorker: Finding BPM for chosen segment of {video_id}")
        bpm_of_segment, _ = mixer_engine.find_bpm_and_beats(y_best_segment, sr_segment)
        
        # The 'tp' (transition point) for this song_data dict will be the actual duration of the selected segment.
        # create_dj_mix will use this 'tp' as the length of y_best_segment.
        tp_sec_for_dict = actual_segment_duration_sec 
        
        print(f"MixGenWorker: Successfully processed {video_id} - Title: {title_from_dl_or_cache}. "
              f"Best segment duration (tp): {tp_sec_for_dict:.2f}s, Segment BPM: {bpm_of_segment:.1f}")
        
        return video_id, {
            'y': y_best_segment,        # This is now the ~1-minute audio data of the chosen segment
            'sr': sr_segment,           # Sample rate of this segment (should be target_sr)
            'bpm': bpm_of_segment,      # BPM calculated from this specific segment
            'tp': tp_sec_for_dict,      # Actual duration of this segment
            'path': mp3_path,           # Original MP3 path (for reference or re-processing if needed)
            'title': title_from_dl_or_cache, # Best available title
            'id': video_id
        }, None # No error message string if successful

    except Exception as e_worker:
        # Catch any other unexpected errors in the worker
        print(f"MixGenWorker: UNHANDLED EXCEPTION processing song {video_id}: {type(e_worker).__name__} - {e_worker}")
        import traceback
        traceback.print_exc() # Print full traceback for detailed debugging
        return video_id, None, str(e_worker)


def mix_generation_background_thread():
    print("App: Mix Generation Background Thread Started.")
    # Initial short delay to allow Flask app to fully start, then first mix generation attempt
    time.sleep(20) 

    while True:
        with mix_state_lock:
            time_until_next_gen = current_mix_state["next_mix_generation_starts_at_ts"] - time.time()
            is_already_processing = current_mix_state["is_server_processing_mix"]

        if is_already_processing: # Avoid overlapping generation cycles
            # print("MixGenThread: Already processing a mix. Waiting...")
            time.sleep(30)
            continue
            
        if time_until_next_gen > 0:
            # print(f"MixGenThread: Sleeping for {min(time_until_next_gen, 30.0):.0f}s until next scheduled generation.")
            time.sleep(min(time_until_next_gen, 30.0)) # Check schedule frequently but don't busy-wait
            continue

        # --- Time to generate a new mix ---
        print(f"App [{time.strftime('%H:%M:%S')}]: Mix Generation Cycle Starting...")
        with mix_state_lock: current_mix_state["is_server_processing_mix"] = True

        selected_ids_for_processing = []
        with votes_lock, played_ids_lock:
            # Get more candidates than needed to filter out played ones
            candidate_votes = votes_counter.most_common(NUM_SONGS_FOR_MIX * 3 + len(played_video_ids_history))
            for vid, count in candidate_votes:
                if count == 0: break # No more songs with votes
                if vid not in played_video_ids_history:
                    selected_ids_for_processing.append(vid)
                    if len(selected_ids_for_processing) >= NUM_SONGS_FOR_MIX:
                        break
            
            # Fallback: If not enough unique unplayed songs, pick *any* voted unplayed songs
            if len(selected_ids_for_processing) < NUM_SONGS_FOR_MIX:
                print(f"App MixGen: Only {len(selected_ids_for_processing)} unplayed top-voted. Trying to fill with other voted unplayed songs.")
                all_voted_unplayed = [vid for vid in votes_counter if vid not in played_video_ids_history and vid not in selected_ids_for_processing]
                random.shuffle(all_voted_unplayed) # Add some variety
                needed_more = NUM_SONGS_FOR_MIX - len(selected_ids_for_processing)
                selected_ids_for_processing.extend(all_voted_unplayed[:needed_more])

        if len(selected_ids_for_processing) < NUM_SONGS_FOR_MIX:
            print(f"App MixGen: Still insufficient unique/unplayed songs ({len(selected_ids_for_processing)} of {NUM_SONGS_FOR_MIX}). Waiting before retry.")
            update_next_mix_schedule_time(processing_just_finished=True) # Reschedule (will add interval from now)
            # No explicit sleep here, the main loop will handle it based on next_mix_generation_starts_at_ts
            continue # Skip to next iteration of the while True loop
        
        print(f"App MixGen: Selected Video IDs for current mix: {selected_ids_for_processing}")

        # Parallel processing of songs
        processed_song_data_map_local = {}
        max_workers = min(NUM_SONGS_FOR_MIX, 1) # Adjust based on CPU/Spleeter performance
        print(f"App MixGen: Starting parallel song processing for {len(selected_ids_for_processing)} songs with {max_workers} workers...")
        
        processing_start_ts = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(song_processing_task_worker, vid_id, 16000): vid_id 
                for vid_id in selected_ids_for_processing
            }
            for future_item in concurrent.futures.as_completed(future_map):
                original_video_id = future_map[future_item]
                try:
                    _, song_data_dict, error_msg_worker = future_item.result()
                    if error_msg_worker:
                        print(f"App MixGen: Error from worker for {original_video_id}: {error_msg_worker}")
                    elif song_data_dict:
                        processed_song_data_map_local[original_video_id] = song_data_dict
                        print(f"App MixGen: Successfully processed data for {original_video_id}")
                except Exception as exc_future:
                    print(f"App MixGen: Exception retrieving result for {original_video_id}: {exc_future}")
        
        processing_duration_secs = time.time() - processing_start_ts
        print(f"App MixGen: Parallel song data processing completed in {processing_duration_secs:.2f} seconds.")

        # Order songs for the mix and collect titles
        final_ordered_song_data_for_mix = []
        current_mix_titles = []
        current_mix_ids_final = []
        for vid_id_ordered in selected_ids_for_processing: # Maintain selection order
            if vid_id_ordered in processed_song_data_map_local:
                song_data = processed_song_data_map_local[vid_id_ordered]
                final_ordered_song_data_for_mix.append(song_data)
                current_mix_titles.append(song_data['title'])
                current_mix_ids_final.append(vid_id_ordered)
            if len(final_ordered_song_data_for_mix) >= NUM_SONGS_FOR_MIX:
                break # Should already be this length if all processed

        if len(final_ordered_song_data_for_mix) < NUM_SONGS_FOR_MIX:
            print(f"App MixGen: Failed to process enough songs for mix ({len(final_ordered_song_data_for_mix)} of {NUM_SONGS_FOR_MIX}). Aborting this mix cycle.")
            update_next_mix_schedule_time(processing_just_finished=True)
            continue

        # Create the actual DJ mix using mixer_engine
        mix_timestamp = int(time.time())
        new_mix_filename_only = f"mix_{mix_timestamp}.mp3"
        new_mix_full_path = os.path.join(GENERATED_MIXES_DIR, new_mix_filename_only)

        print(f"App MixGen: Creating final mix with {len(final_ordered_song_data_for_mix)} songs -> {new_mix_full_path}")
        
        generated_mix_filepath = mixer_engine.create_dj_mix(
            final_ordered_song_data_for_mix,
            new_mix_full_path,
            crossfade_duration_ms=CROSSFADE_MS,
            tempo_match_window_sec=8.0 # Assuming you want to keep this default or make it configurable
        )

        if generated_mix_filepath:
            print(f"App MixGen: New mix generated: {generated_mix_filepath}")
            with mix_state_lock, played_ids_lock, votes_lock:
                current_mix_state["url"] = f"/{os.path.basename(GENERATED_MIXES_DIR)}/{new_mix_filename_only}"
                current_mix_state["songs_in_mix_titles"] = current_mix_titles
                current_mix_state["songs_in_mix_ids"] = current_mix_ids_final
                current_mix_state["generated_at_ts"] = mix_timestamp
                
                for vid_just_played in current_mix_ids_final:
                    if vid_just_played not in played_video_ids_history: # deque handles maxlen
                        played_video_ids_history.append(vid_just_played)
                    if vid_just_played in votes_counter:
                        del votes_counter[vid_just_played] # Reset votes for songs just played
            
            cleanup_old_generated_mixes()
        else:
            print(f"App MixGen: Mix creation failed in mixer_engine.")
        
        update_next_mix_schedule_time(processing_just_finished=True)
        print(f"App MixGen: Cycle Complete. Next mix generation scheduled around {time.ctime(current_mix_state['next_mix_generation_starts_at_ts'])}")
        # Loop will handle waiting for the next scheduled time.


# --- UTILITY/CLEANUP FUNCTIONS ---
def cleanup_old_generated_mixes():
    try:
        mix_files_on_disk = [
            os.path.join(GENERATED_MIXES_DIR, f) 
            for f in os.listdir(GENERATED_MIXES_DIR) 
            if f.startswith("mix_") and f.endswith(".mp3")
        ]
        # Sort by modification time (oldest first)
        mix_files_on_disk.sort(key=os.path.getmtime)
        
        while len(mix_files_on_disk) > MAX_GENERATED_MIXES_TO_KEEP:
            oldest_mix_to_remove = mix_files_on_disk.pop(0) # Remove from list
            os.remove(oldest_mix_to_remove) # Delete from disk
            print(f"App Cleanup: Removed old mix file: {oldest_mix_to_remove}")
    except Exception as e:
        print(f"App Cleanup: Error during old mix cleanup: {e}")


# --- MAIN APP EXECUTION ---
if __name__ == '__main__':
    initialize_app_components()
    # Start the background mix generation thread
    background_mix_thread = threading.Thread(target=mix_generation_background_thread, daemon=True)
    background_mix_thread.start()
    # Run Flask app (debug=False for production/threading stability)
    app.run(host='0.0.0.0', port=5000, debug=False)