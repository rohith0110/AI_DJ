// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element Cache ---
    const ui = {
        statusBar: document.getElementById('system-status-bar'),
        searchForm: document.getElementById('song-search-form'),
        queryInput: document.getElementById('song-query-input'),
        searchButton: document.getElementById('search-song-button'),
        searchResultsArea: document.getElementById('search-results-display-area'),
        voteMessageFeedback: document.getElementById('user-voting-message'),
        currentMixTracklist: document.getElementById('current-mix-tracklist'),
        mainPlayer: document.getElementById('main-audio-player'),
        preloadPlayer: document.getElementById('preload-audio-player'),
        nextMixTimer: document.getElementById('next-mix-timer-span'),
        serverActivityIndicator: document.getElementById('server-activity-indicator'),
        topVotedList: document.getElementById('top-voted-preview-list')
    };

    // --- Application State ---
    let activePlayer = ui.mainPlayer;
    let inactivePlayer = ui.preloadPlayer;
    let currentLoadedMixUrlPath = null; // Store only pathname for comparison
    const CROSSFADE_DURATION_SEC = 2.5;
    let statusUpdateIntervalId = null;
    let searchDebounceTimeout = null;

    // --- Event Listeners ---
    ui.searchForm.addEventListener('submit', handleSearchSubmit);
    ui.queryInput.addEventListener('input', handleSearchInputDebounce); // Debounce searches

    // Audio player error handling
    [ui.mainPlayer, ui.preloadPlayer].forEach(player => {
        player.addEventListener('error', (e) => {
            console.error(`Audio player error on ${player.id}:`, e.target.error);
            ui.statusBar.textContent = `Stream error. Please wait or refresh.`;
        });
        player.addEventListener('ended', handleActivePlayerEnded);
    });
    
    // Attempt to resume audio context on user interaction if suspended by browser
    function tryResumeAudioContext() {
        if (activePlayer.paused && activePlayer.readyState >= 2 && activePlayer.src) {
            activePlayer.play().catch(e => console.warn("Autoplay/resume play failed on interaction:", e));
        }
        document.body.removeEventListener('click', tryResumeAudioContext);
        document.body.removeEventListener('keydown', tryResumeAudioContext);
    }
    document.body.addEventListener('click', tryResumeAudioContext, { once: true });
    document.body.addEventListener('keydown', tryResumeAudioContext, { once: true });


    // --- Core Functions ---
    function handleSearchInputDebounce() {
        clearTimeout(searchDebounceTimeout);
        const query = ui.queryInput.value.trim();
        if (query.length < 3) { // Minimum query length
            ui.searchResultsArea.innerHTML = ''; // Clear if query too short
            return;
        }
        searchDebounceTimeout = setTimeout(() => {
            // Trigger form submission programmatically after debounce
             ui.searchForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        }, 400); // 400ms debounce time
    }

    async function handleSearchSubmit(event) {
        if(event) event.preventDefault(); // Prevent actual form submission if triggered by event
        const query = ui.queryInput.value.trim();
        if (query.length < 3) return; // Already handled by debounce, but good check

        ui.searchButton.disabled = true;
        ui.searchButton.textContent = 'Searching...';
        ui.searchResultsArea.innerHTML = '<p class="message-feedback info">Finding tracks...</p>';

        try {
            const response = await fetch('/search_songs', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query})
            });
            const searchData = await response.json();
            renderSearchResults(searchData);
        } catch (error) {
            console.error('Search request failed:', error);
            ui.searchResultsArea.innerHTML = '<p class="message-feedback error">Search failed. Please check connection.</p>';
        } finally {
            ui.searchButton.disabled = false;
            ui.searchButton.textContent = 'Search';
        }
    }

    function renderSearchResults(results) {
        ui.searchResultsArea.innerHTML = '';
        if (!results || results.error) {
            ui.searchResultsArea.innerHTML = `<p class="message-feedback error">${results.error || 'No results or error during search.'}</p>`;
            return;
        }
        if (results.length === 0) {
            ui.searchResultsArea.innerHTML = '<p class="message-feedback info">No tracks found for that query.</p>';
            return;
        }
        const ul = document.createElement('ul');
        results.forEach(song => {
            const li = document.createElement('li');
            li.className = 'search-result-entry';
            li.innerHTML = `
                <img src="${song.thumbnail_url || 'https://via.placeholder.com/45'}" alt="Art">
                <span class="song-info-title">${song.title}</span>
                <button data-video-id="${song.video_id}" class="vote-button">Vote</button>
            `;
            li.querySelector('.vote-button').addEventListener('click', (e) => {
                castVote(song.video_id, e.target);
            });
            ul.appendChild(li);
        });
        ui.searchResultsArea.appendChild(ul);
    }

    async function castVote(videoId, buttonEl) {
        if (buttonEl) {
            buttonEl.disabled = true;
            buttonEl.textContent = 'Voting...';
        }
        try {
            const response = await fetch('/submit_vote', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({video_id: videoId})
            });
            const voteResult = await response.json();
            displayFeedbackMessage(voteResult.message, voteResult.success ? 'success' : 'error');
            if (voteResult.success) {
                fetchServerStatus();
                 // Optionally clear search results after a successful vote
                 // setTimeout(() => { ui.searchResultsArea.innerHTML = ''; }, 1500);
            }
        } catch (error) {
            console.error('Vote submission failed:', error);
            displayFeedbackMessage('Vote submission error. Try again.', 'error');
        } finally {
            if (buttonEl) { // Re-enable button unless it was a rate limit hit message
                if (!ui.voteMessageFeedback.textContent.toLowerCase().includes("limit reached")) {
                    setTimeout(() => {
                        buttonEl.disabled = false;
                        buttonEl.textContent = 'Vote';
                    }, 1000);
                }
            }
        }
    }

    function displayFeedbackMessage(message, type = 'info') {
        ui.voteMessageFeedback.textContent = message;
        ui.voteMessageFeedback.className = `message-feedback ${type}`;
        setTimeout(() => {
            ui.voteMessageFeedback.textContent = '';
            ui.voteMessageFeedback.className = 'message-feedback';
        }, 6000); // Message visible for 6 seconds
    }
    
    function formatTimeCountdown(seconds) {
        if (isNaN(seconds) || seconds < 0) return '00:00';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    function updateUIDisplays(statusData) {
        // System Status Bar
        if (statusData.current_mix_url && statusData.current_mix_generated_at_ts) {
            const mixDate = new Date(statusData.current_mix_generated_at_ts * 1000);
            ui.statusBar.textContent = `Live Mix Active (Generated: ${mixDate.toLocaleTimeString()})`;
        } else if (statusData.is_server_processing_mix) {
            ui.statusBar.textContent = "Hold tight! Server is crafting the next awesome mix...";
        } else {
            ui.statusBar.textContent = "Waiting for the DJ to spin the first tracks...";
        }

        // Current Mix Tracklist
        ui.topVotedList.innerHTML = '';
        if (statusData.top_voted_preview && statusData.top_voted_preview.length > 0) {
            statusData.top_voted_preview.forEach(song => {
                const li = document.createElement('li');
        li.innerHTML = `${song.title} <span class="vote-count-text">(Votes: ${song.votes})</span>`;
                ui.topVotedList.appendChild(li);
            });
        } else {
            ui.topVotedList.innerHTML = '<li>No active votes yet for upcoming tracks.</li>';
        }

        // Next Mix Countdown
        ui.nextMixTimer.textContent = formatTimeCountdown(statusData.next_mix_generation_starts_in_seconds);
        
        // Server Processing Indicator
        ui.serverActivityIndicator.style.display = statusData.is_server_processing_mix ? 'block' : 'none';

        // Top Voted Songs Preview
        ui.topVotedList.innerHTML = '';
        if (statusData.top_voted_preview && statusData.top_voted_preview.length > 0) {
            statusData.top_voted_preview.forEach(song => {
                const li = document.createElement('li');
                li.innerHTML = `${song.title} <span class="vote-count-text">(Votes: ${song.votes})</span>`;
                ui.topVotedList.appendChild(li);
            });
        } else {
            ui.topVotedList.innerHTML = '<li>No active votes yet for upcoming tracks.</li>';
        }
    }

    function handleActivePlayerEnded() {
        console.log("Active player finished. Current behavior: stops. (Could prefetch/auto-transition if desired)");
        // If you want continuous play even if server is slow, you'd need logic here to
        // check if inactivePlayer has the *next* segment of the *same* mix, or a new mix.
        // For now, it just ends. The server should ideally provide a new mix before this.
    }
    
    async function manageAudioStream(newMixUrlFromServer) {
        if (!newMixUrlFromServer) {
            if (currentLoadedMixUrlPath) { // If something was playing/loaded
                console.log("Server reports no current mix. Fading out audio.");
                audioFade(activePlayer, 0, CROSSFADE_DURATION_SEC, () => { activePlayer.pause(); activePlayer.src = ''; });
                audioFade(inactivePlayer, 0, 0.1, () => { inactivePlayer.pause(); inactivePlayer.src = ''; }); // Quick stop
                currentLoadedMixUrlPath = null;
            }
            return;
        }

        const newMixPath = new URL(newMixUrlFromServer, window.location.origin).pathname;

        if (newMixPath === currentLoadedMixUrlPath) {
            // console.log("New mix URL path is the same as current. No change needed.");
            return; 
        }

        console.log(`New mix detected: ${newMixPath}. Current: ${currentLoadedMixUrlPath || 'None'}. Preparing switch.`);
        
        // The new target for what should be playing or loading
        const newTargetUrl = newMixUrlFromServer; 

        // Assign new source to the inactive player and start loading
        inactivePlayer.src = newTargetUrl;
        inactivePlayer.load();

        try {
            await new Promise((resolve, reject) => {
                inactivePlayer.oncanplaythrough = resolve;
                inactivePlayer.onerror = (e) => reject(new Error(`Inactive player (${inactivePlayer.id}) load error.`));
                setTimeout(() => reject(new Error("Timeout waiting for inactive player to load new mix.")), 20000); // 20s
            });

            console.log("Inactive player ready with new mix. Starting crossfade.");
            
            // Mute new player, start playback (browsers might block this until interaction)
            inactivePlayer.volume = 0;
            inactivePlayer.play().catch(e => console.warn("Autoplay on inactive player (for crossfade) was prevented:", e));
            
            // Fade out currently active player
            if (activePlayer.src && !activePlayer.paused) {
                audioFade(activePlayer, 0, CROSSFADE_DURATION_SEC, () => {
                    activePlayer.pause();
                    // activePlayer.src = ''; // Don't clear src immediately, could be reused if next load fails
                    activePlayer.currentTime = 0;
                });
            } else {
                activePlayer.pause();
                activePlayer.currentTime = 0;
            }

            // Fade in the (previously) inactive player
            audioFade(inactivePlayer, 1, CROSSFADE_DURATION_SEC);

            // Swap player roles
            const tempPlayer = activePlayer;
            activePlayer = inactivePlayer;
            inactivePlayer = tempPlayer;
            currentLoadedMixUrlPath = newMixPath; // Update with the new successfully loaded path

        } catch (error) {
            console.error("Error during mix transition (preload/crossfade):", error);
            // Fallback: Hard switch to the new URL on the *current* active player
            console.log("Fallback: Hard switching main player to new mix URL.");
            activePlayer.src = newTargetUrl;
            activePlayer.load();
            activePlayer.volume = 1;
            activePlayer.play().catch(e => console.warn("Autoplay for fallback hard switch prevented:", e));
            
            inactivePlayer.pause(); // Ensure other player is stopped
            inactivePlayer.src = '';
            currentLoadedMixUrlPath = newMixPath;
        }
    }

    function audioFade(player, targetVolume, durationSec, onComplete) {
        const startVolume = player.volume;
        const DURATION_MS = durationSec * 1000;
        if (DURATION_MS <= 0) { // Instant change
            player.volume = targetVolume;
            if (targetVolume === 0 && player.src) player.pause();
            if (onComplete) onComplete();
            return;
        }
        const startTime = performance.now();

        function animate() {
            const elapsedTime = performance.now() - startTime;
            const progress = Math.min(elapsedTime / DURATION_MS, 1);
            player.volume = startVolume + (targetVolume - startVolume) * progress;

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                player.volume = targetVolume;
                if (targetVolume === 0 && player.src) { // Only pause if it has a source
                    player.pause();
                }
                if (onComplete) onComplete();
            }
        }
        if (targetVolume > 0 && player.paused && player.src) { // If fading in and paused, try to play
            player.play().catch(e => console.warn("Playback attempt during fade-in prevented:", e));
        }
        requestAnimationFrame(animate);
    }

    async function fetchServerStatus() {
        try {
            const response = await fetch('/current_status');
            if (!response.ok) {
                throw new Error(`Status fetch failed: ${response.status}`);
            }
            const statusData = await response.json();
            updateUIDisplays(statusData);
            manageAudioStream(statusData.current_mix_url); // Pass the full URL
        } catch (error) {
            console.error('Failed to fetch or process server status:', error);
            ui.statusBar.textContent = 'Error connecting to server. Retrying...';
            // Potentially stop audio or show error more prominently
        }
    }
    
    // --- Initial Load & Periodic Updates ---
    console.log("AI DJ Client Initialized. Waiting for first status update...");
    fetchServerStatus(); // Initial call
    statusUpdateIntervalId = setInterval(fetchServerStatus, 7500); // Poll every 7.5 seconds
});