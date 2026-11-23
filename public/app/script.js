/**
 * InterviewAce — AI-Powered Mock Interview & Placement Assistant
 * Frontend Controller & Voice/Turn Manager
 */

let sessionId = null;
let recognition = null;
let synth = window.speechSynthesis;
let isListening = false;
let resumeText = "";
let interviewStartTime = null;
let timerInterval = null;
let shouldKeepListening = false;
let accumulatedTranscript = '';
let resumeWarnings = [];
let jobDescription = "";

let totalQuestions = 5;
let currentQuestionIndex = 1;
let sessionTotalScore = 0;
let answeredQuestionsCount = 0;
let evaluationsHistory = [];
let questionScores = {};
let isStartingSession = false;
let isSubmittingResponse = false;
let isPasteDetected = false;
let isGeneratingReport = false;
let isInterviewCompleted = false;
let activeScreen = "dashboard";

const API_URL = "";

// Initialize tip, defaults, paste listeners, and dashboard on page load
document.addEventListener("DOMContentLoaded", () => {
    onRoleChange();
    initPasteDetection();
    initWordCounter();
    renderDashboard();
});

// ==============================================================================
// Screen Navigation & Sidebar Controls
// ==============================================================================

function toggleSidebar() {
    const sidebar = document.getElementById("app-sidebar");
    if (sidebar) {
        sidebar.classList.toggle("collapsed");
        sidebar.classList.toggle("mobile-open");
    }
}

function navigateTo(screenId) {
    activeScreen = screenId;

    // Screens map
    const screens = {
        "dashboard": document.getElementById("dashboard-screen"),
        "setup": document.getElementById("setup-screen"),
        "interview": document.getElementById("interview-screen"),
        "feedback": document.getElementById("feedback-screen"),
        "history": document.getElementById("history-screen"),
        "analytics": document.getElementById("analytics-screen"),
        "rubric-info": document.getElementById("rubric-info-screen"),
    };

    // Hide all screens
    Object.values(screens).forEach(s => {
        if (s) s.classList.add("hidden");
    });

    // Show selected screen
    if (screens[screenId]) {
        screens[screenId].classList.remove("hidden");
    }

    // Update navigation active states
    document.querySelectorAll(".nav-item").forEach(btn => btn.classList.remove("active"));
    const activeNavBtn = document.getElementById(`nav-${screenId}-btn`);
    if (activeNavBtn) activeNavBtn.classList.add("active");

    // Update Header Breadcrumb Title
    const titleMap = {
        "dashboard": "Dashboard",
        "setup": "Configure Interview",
        "interview": "Live Mock Interview",
        "feedback": "Performance Report",
        "history": "Practice History",
        "analytics": "Performance Analytics",
        "rubric-info": "4-Criteria Rubric Guide",
    };
    const titleEl = document.getElementById("current-screen-title");
    if (titleEl) {
        titleEl.textContent = titleMap[screenId] || "Dashboard";
    }

    // Dynamic screen data renderers
    if (screenId === "dashboard") {
        renderDashboard();
    } else if (screenId === "history") {
        renderHistoryScreen();
    } else if (screenId === "analytics") {
        renderAnalyticsScreen();
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
}

function startQuickPreset(role, track, difficulty) {
    const roleSelect = document.getElementById("role-select");
    const typeSelect = document.getElementById("type-select");
    const diffSelect = document.getElementById("difficulty-select");

    if (roleSelect) roleSelect.value = role;
    if (typeSelect) typeSelect.value = track;
    if (diffSelect) diffSelect.value = difficulty;

    onRoleChange();
    navigateTo("setup");
}

// ==============================================================================
// Paste Detection and Integrity Controls
// ==============================================================================

function markPasteDetected() {
    isPasteDetected = true;
    const banner = document.getElementById("paste-warning-banner");
    if (banner) {
        banner.classList.remove("hidden");
    }
    const badge = document.getElementById("integrity-badge");
    if (badge) {
        badge.textContent = "Pasted Content ⚠️";
        badge.style.color = "var(--warning)";
    }
}

function resetPasteDetection() {
    isPasteDetected = false;
    const banner = document.getElementById("paste-warning-banner");
    if (banner) {
        banner.classList.add("hidden");
    }
    const badge = document.getElementById("integrity-badge");
    if (badge) {
        badge.textContent = "Original Input ✓";
        badge.style.color = "var(--success)";
    }
}

function initPasteDetection() {
    const textInput = document.getElementById("text-answer-input");
    if (!textInput) return;

    // 1. Browser native paste event (covers right-click -> Paste, browser edit menu, middle-click paste)
    textInput.addEventListener("paste", () => {
        markPasteDetected();
    });

    // 2. Keyboard shortcuts: Ctrl+V, Cmd+V, Shift+Insert
    textInput.addEventListener("keydown", (e) => {
        const isCtrlOrCmdV = (e.ctrlKey || e.metaKey) && (e.key === "v" || e.key === "V");
        const isShiftInsert = e.shiftKey && (e.key === "Insert" || e.keyCode === 45);
        if (isCtrlOrCmdV || isShiftInsert) {
            markPasteDetected();
        }
    });

    // 3. Drag and drop external text
    textInput.addEventListener("drop", () => {
        markPasteDetected();
    });

    // 4. Reset warning when input is emptied
    textInput.addEventListener("input", () => {
        if (textInput.value.trim().length === 0) {
            resetPasteDetection();
        }
    });
}

function initWordCounter() {
    const textInput = document.getElementById("text-answer-input");
    if (!textInput) return;
    textInput.addEventListener("input", updateWordCount);
}

function updateWordCount() {
    const textInput = document.getElementById("text-answer-input");
    const countEl = document.getElementById("answer-char-count");
    if (!textInput || !countEl) return;
    const text = textInput.value.trim();
    const words = text.length === 0 ? 0 : text.split(/\s+/).length;
    const chars = textInput.value.length;
    countEl.textContent = `${words} words • ${chars} chars`;
}

// Dynamic tip update based on selected role
function onRoleChange() {
    const roleSelect = document.getElementById("role-select");
    if (!roleSelect) return;
    const role = roleSelect.value;
    const tipEl = document.getElementById("dynamic-tip");
    if (!tipEl) return;

    if (role.includes("Software") || role.includes("Engineer") || role.includes("Backend")) {
        tipEl.textContent = "For technical coding, first explain your brute-force logic, state time/space complexity O(N), then optimize using appropriate data structures.";
    } else if (role.includes("Frontend")) {
        tipEl.textContent = "Highlight DOM manipulation, UI responsiveness, CSS specificity, accessibility (a11y), and state management tradeoffs.";
    } else if (role.includes("Data Analyst") || role.includes("Scientist")) {
        tipEl.textContent = "For SQL and analysis, mention specific JOIN conditions, aggregations, window functions, and business impact of the metrics.";
    } else if (role.includes("Machine Learning")) {
        tipEl.textContent = "Clarify feature engineering, loss functions, precision vs recall trade-offs, and how you detect/prevent overfitting.";
    } else if (role.includes("HR")) {
        tipEl.textContent = "Structure behavioral responses using the STAR method: Situation, Task, Action, and measurable Result.";
    } else {
        tipEl.textContent = "Be concise, ground your answers in concrete examples from your past projects, and avoid unnecessary buzzwords.";
    }
}

function onPromptVersionChange() {
    const ver = document.getElementById("prompt-version-select").value.toUpperCase();
    const tag = document.getElementById("prompt-tag");
    if (tag) tag.textContent = `Prompt ${ver} (Active)`;
}

// Timer Functions
function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - interviewStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        const timerEl = document.getElementById('timer-display');
        if (timerEl) {
            timerEl.textContent = `⏱️ ${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// Switch between Typed Answer and Voice Input
function switchInputMode(mode) {
    const textTab = document.getElementById("mode-text-btn");
    const voiceTab = document.getElementById("mode-voice-btn");
    const textArea = document.getElementById("text-mode-area");
    const voiceArea = document.getElementById("voice-mode-area");

    if (mode === 'text') {
        textTab.classList.add("active");
        voiceTab.classList.remove("active");
        textArea.classList.remove("hidden");
        voiceArea.classList.add("hidden");
        stopListening();
    } else {
        voiceTab.classList.add("active");
        textTab.classList.remove("active");
        voiceArea.classList.remove("hidden");
        textArea.classList.add("hidden");
    }
}

// Target Role Detection
async function detectRole() {
    const goal = document.getElementById("goal-input").value.trim();
    const btn = document.getElementById("detect-btn");
    const resumeFile = document.getElementById("resume-upload").files[0];

    if (!goal && !resumeFile) {
        alert("Please enter what you want to practice for, or upload a resume first.");
        return;
    }

    btn.disabled = true;
    btn.textContent = "Detecting...";

    try {
        if (resumeFile && !resumeText) {
            await uploadResume(resumeFile);
        }

        const response = await fetch(`${API_URL}/detect_role`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ resume_text: resumeText || null, goal_statement: goal || null })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        showDetection(await response.json());
    } catch (err) {
        console.error("Role detection failed", err);
        alert("Could not detect role: " + err.message + "\nPlease select from the dropdown.");
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>✨ Detect Role</span>";
    }
}

function showDetection(detection) {
    const box = document.getElementById("detection-result");
    document.getElementById("detected-role").textContent = detection.role;

    const conf = document.getElementById("detected-confidence");
    conf.textContent = detection.confidence + " confidence";
    conf.className = "confidence confidence-" + detection.confidence;

    document.getElementById("detected-reasoning").textContent = detection.reasoning || "";
    box.classList.remove("hidden");

    // Pre-select in dropdown
    const roleSelect = document.getElementById("role-select");
    if ([...roleSelect.options].some(o => o.value === detection.role)) {
        roleSelect.value = detection.role;
        onRoleChange();
    }

    if (detection.suggested_experience_level) {
        const expSelect = document.getElementById("experience-select");
        if ([...expSelect.options].some(o => o.value === detection.suggested_experience_level)) {
            expSelect.value = detection.suggested_experience_level;
        }
    }
}

// Resume Upload Handler
async function uploadResume(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_URL}/upload_resume`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Resume upload failed (HTTP ${res.status})`);

    const data = await res.json();
    resumeText = data.resume_text || "";
    resumeWarnings = data.warnings || [];

    const note = document.getElementById("file-name");
    if (note) {
        if (resumeWarnings.length) {
            note.textContent = "⚠️ " + resumeWarnings.join(" ");
            note.style.color = "var(--warning)";
        } else {
            note.textContent = "✓ Resume parsed successfully (" + file.name + ")";
            note.style.color = "var(--success)";
        }
    }
    return data;
}

function updateFileName(input) {
    const file = input.files[0];
    if (file) {
        document.getElementById('file-name').textContent = "Selected: " + file.name;
    }
}

// Start Interview Session
async function startInterview() {
    if (isStartingSession) return;
    isStartingSession = true;

    const startBtn = document.querySelector(".start-btn");
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerHTML = "<span>Starting Session... ⏳</span>";
    }

    // Clear previous conversation to prevent duplicated messages
    const chatHistory = document.getElementById("chat-history");
    if (chatHistory) chatHistory.innerHTML = "";

    const role = document.getElementById("role-select").value;
    const interviewType = document.getElementById("type-select").value;
    const difficulty = document.getElementById("difficulty-select").value;
    const experience = document.getElementById("experience-select").value;
    const questionCount = parseInt(document.getElementById("count-select").value, 10) || 5;
    const promptVersion = document.getElementById("prompt-version-select").value;
    const resumeFile = document.getElementById("resume-upload").files[0];
    jobDescription = document.getElementById("jd-input").value.trim();

    totalQuestions = questionCount;
    currentQuestionIndex = 1;
    sessionTotalScore = 0;
    answeredQuestionsCount = 0;
    evaluationsHistory = [];
    questionScores = {};
    isInterviewCompleted = false;
    isGeneratingReport = false;
    resetPasteDetection();

    const endBtn = document.getElementById("end-btn");
    if (endBtn) {
        endBtn.disabled = false;
        endBtn.innerHTML = "<span>Finish Interview &amp; Get Report 📊</span>";
    }
    const submitBtn = document.getElementById("submit-text-btn");
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = "<span>📤 Submit Answer for Evaluation</span>";
    }
    const textInput = document.getElementById("text-answer-input");
    if (textInput) {
        textInput.disabled = false;
        textInput.placeholder = "Type your code, algorithm approach, or explanation here...\n\nTip: Structure your thoughts clearly. If coding, state approach, write code, then explain time/space complexity.";
        textInput.value = "";
    }
    updateWordCount();

    updateStatus("Initializing InterviewAce Session...");

    if (resumeFile && !resumeText) {
        try {
            updateStatus("Parsing Resume...");
            await uploadResume(resumeFile);
        } catch (err) {
            console.error("Resume upload failed", err);
        }
    }

    try {
        updateStatus("Starting Session...");
        const response = await fetch(`${API_URL}/start_session`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                role,
                interview_type: interviewType,
                difficulty,
                experience_level: experience,
                question_count: questionCount,
                prompt_version: promptVersion,
                resume_text: resumeText || null
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        sessionId = data.session_id;

        // Update Topbar UI
        document.getElementById("interview-role-badge").textContent = role;
        document.getElementById("interview-track-badge").textContent = interviewType + " Track";
        document.getElementById("interview-diff-badge").textContent = difficulty;
        document.getElementById("interview-prompt-badge").textContent = `Prompt ${promptVersion.toUpperCase()}`;

        currentQuestionIndex = 0;
        document.getElementById("q-progress-text").textContent = "Introduction";
        document.getElementById("running-score-text").textContent = `0 / ${totalQuestions * 20}`;
        const currentAnswerEl = document.getElementById("current-answer-score-text");
        if (currentAnswerEl) currentAnswerEl.textContent = "-- / 20";

        navigateTo("interview");

        interviewStartTime = Date.now();
        startTimer();

        addMessage("agent", data.initial_message);
        speak(data.initial_message);
    } catch (err) {
        console.error(err);
        alert("Could not start session: " + err.message);
        updateStatus("Ready");
    } finally {
        isStartingSession = false;
        if (startBtn) {
            startBtn.disabled = false;
            startBtn.innerHTML = "<span>Start Interview Session 🚀</span>";
        }
    }
}

// Conversation Stream Management
function addMessage(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.textContent = role === "agent" ? "🤖" : "👤";
    bubble.appendChild(avatar);

    const body = document.createElement("div");
    body.className = "msg-body";

    const author = document.createElement("div");
    author.className = "msg-author";
    author.textContent = role === "agent" ? "Interviewer" : "You (Candidate)";
    if (role === "agent") {
        const replayBtn = document.createElement("button");
        replayBtn.className = "replay-voice-btn";
        replayBtn.title = "Replay audio for this question";
        replayBtn.innerHTML = "🔊 Speak";
        replayBtn.onclick = (e) => {
            e.stopPropagation();
            speak(text);
        };
        author.appendChild(replayBtn);
    }
    body.appendChild(author);

    const content = document.createElement("div");
    content.className = "msg-text";
    content.textContent = text;
    body.appendChild(content);

    bubble.appendChild(body);

    const history = document.getElementById("chat-history");
    if (history) {
        history.appendChild(bubble);
        history.scrollTop = history.scrollHeight;
    }
}

// Render the Fixed 4-Criteria Rubric Card right in the chat stream
function renderRubricCard(evaluation) {
    if (!evaluation || !evaluation.rubric) return;

    const r = evaluation.rubric;
    const card = document.createElement("div");
    card.className = "rubric-eval-card";

    // Header with total score & percentage
    const head = document.createElement("div");
    head.className = "eval-header";

    const title = document.createElement("div");
    title.className = "eval-title";
    title.innerHTML = "<span>📋 Answer Evaluation (Fixed Rubric)</span>";
    head.appendChild(title);

    const totalPill = document.createElement("div");
    totalPill.className = "eval-total-pill";
    totalPill.textContent = `${r.total_score} / 20 (${r.percentage}%)`;
    head.appendChild(totalPill);

    card.appendChild(head);

    // Paste Alert Banner if detected
    if (evaluation.paste_detected || (evaluation.areas_for_improvement && evaluation.areas_for_improvement.some(a => String(a).toLowerCase().includes("paste")))) {
        const pasteAlert = document.createElement("div");
        pasteAlert.className = "paste-eval-alert";
        pasteAlert.innerHTML = "⚠️ <strong>Pasted answer detected.</strong> Please answer in your own words to receive a score.";
        card.appendChild(pasteAlert);
    }

    // 4 Criteria Breakdown Grid
    const grid = document.createElement("div");
    grid.className = "eval-criteria-grid";

    const criteriaList = [
        { name: "Correctness", score: r.correctness },
        { name: "Relevance", score: r.relevance },
        { name: "Clarity", score: r.clarity },
        { name: "Completeness", score: r.completeness },
    ];

    criteriaList.forEach(c => {
        const item = document.createElement("div");
        item.className = "crit-item";

        const nameSpan = document.createElement("span");
        nameSpan.className = "crit-name";
        nameSpan.textContent = c.name;
        item.appendChild(nameSpan);

        const barWrap = document.createElement("div");
        barWrap.className = "crit-score-bar-wrap";

        const barBg = document.createElement("div");
        barBg.className = "crit-bar-bg";
        const barFill = document.createElement("div");
        barFill.className = "crit-bar-fill";
        barFill.style.width = `${(c.score / 5) * 100}%`;
        barBg.appendChild(barFill);
        barWrap.appendChild(barBg);

        const scoreSpan = document.createElement("span");
        scoreSpan.className = "crit-score-num";
        scoreSpan.textContent = `${c.score}/5`;
        barWrap.appendChild(scoreSpan);

        item.appendChild(barWrap);
        grid.appendChild(item);
    });

    card.appendChild(grid);

    // Strengths and Areas for Improvement
    const detailsRow = document.createElement("div");
    detailsRow.className = "eval-details-row";

    const strDiv = document.createElement("div");
    strDiv.className = "eval-strengths";
    strDiv.innerHTML = "<strong>✓ Strengths:</strong>";
    const strUl = document.createElement("ul");
    (evaluation.strengths || ["Good attempt"]).forEach(s => {
        const li = document.createElement("li");
        li.textContent = s;
        strUl.appendChild(li);
    });
    strDiv.appendChild(strUl);
    detailsRow.appendChild(strDiv);

    const impDiv = document.createElement("div");
    impDiv.className = "eval-improvements";
    impDiv.innerHTML = "<strong>⚡ Areas for Improvement:</strong>";
    const impUl = document.createElement("ul");
    (evaluation.areas_for_improvement || ["Elaborate on edge cases"]).forEach(i => {
        const li = document.createElement("li");
        li.textContent = i;
        impUl.appendChild(li);
    });
    impDiv.appendChild(impUl);
    detailsRow.appendChild(impDiv);

    card.appendChild(detailsRow);

    // Suggested Improved Exemplar Answer
    if (evaluation.suggested_improved_answer) {
        const exBox = document.createElement("div");
        exBox.className = "exemplar-answer-box";
        exBox.innerHTML = `
            <div class="exemplar-header">💡 Suggested Exemplar Answer:</div>
            <div class="exemplar-text">"${escapeHtml(evaluation.suggested_improved_answer)}"</div>
        `;
        card.appendChild(exBox);
    }

    const history = document.getElementById("chat-history");
    if (history) {
        history.appendChild(card);
        history.scrollTop = history.scrollHeight;
    }
}

function escapeHtml(str) {
    return (str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function updateStatus(text) {
    const statusEl = document.getElementById("status-text");
    if (statusEl) statusEl.textContent = text;

    const dot = document.getElementById("pulse-dot");
    if (!dot) return;

    if (text.includes("Listening")) {
        dot.className = "pulse-dot listening";
    } else if (text.includes("Speaking") || text.includes("Evaluating") || text.includes("Thinking")) {
        dot.className = "pulse-dot active";
    } else {
        dot.className = "pulse-dot";
    }
}

// Speech Synthesis
const SPEECH_WATCHDOG_GRACE_MS = 5000;
const SPEECH_MS_PER_CHAR = 90;

function speak(text, onDone) {
    if (!('speechSynthesis' in window) || !window.speechSynthesis) {
        if (onDone) onDone();
        return;
    }
    const currentSynth = window.speechSynthesis;
    if (currentSynth.speaking) currentSynth.cancel();

    const cleanText = (text || "")
        .replace(/```[\s\S]*?```/g, "Code block omitted.")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/\*\*([^*]+)\*\*/g, "$1")
        .replace(/\*([^*]+)\*/g, "$1")
        .replace(/#{1,6}\s+/g, "")
        .trim();

    if (!cleanText) {
        if (onDone) onDone();
        return;
    }

    const next = onDone || (() => {
        // Automatically start listening only if in voice mode
        const voiceArea = document.getElementById("voice-mode-area");
        if (voiceArea && !voiceArea.classList.contains("hidden")) {
            startListening();
        } else {
            updateStatus("Awaiting your answer...");
        }
    });

    let handedOff = false;
    let watchdog = null;

    const handOff = (reason) => {
        if (handedOff) return;
        handedOff = true;
        if (watchdog) clearTimeout(watchdog);
        next();
    };

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';
    window._activeUtterance = utterance;

    utterance.onstart = () => updateStatus("Interviewer Speaking...");
    utterance.onend = () => handOff("ended");
    utterance.onerror = (e) => handOff(e.error || "error");

    watchdog = setTimeout(
        () => handOff("watchdog"),
        SPEECH_WATCHDOG_GRACE_MS + cleanText.length * SPEECH_MS_PER_CHAR
    );

    try {
        currentSynth.speak(utterance);
    } catch (err) {
        console.error("SpeechSynthesis error:", err);
        handOff("threw");
    }
}

// Web Speech API for Spoken Answers
function toggleVoiceRecognition() {
    if (isListening) {
        stopListening();
    } else {
        startListening();
    }
}

function startListening() {
    if (isListening) {
        console.log("Speech recognition is already running.");
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("Web Speech API is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
        return;
    }

    if (!recognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            if (event.error === 'aborted') return;
            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                shouldKeepListening = false;
                isListening = false;
                const mic = document.getElementById("mic-icon");
                if (mic) mic.classList.remove("recording");
                const stopBtn = document.getElementById("stop-listening-btn");
                if (stopBtn) stopBtn.style.display = "none";
                updateStatus("Microphone access denied. Please allow microphone permissions or switch to typed mode.");
                const liveEl = document.getElementById("live-transcript");
                if (liveEl) liveEl.textContent = "Microphone access denied. Please allow mic permissions in browser or use Typed Mode.";
                return;
            }
            if (event.error === 'audio-capture') {
                shouldKeepListening = false;
                isListening = false;
                const mic = document.getElementById("mic-icon");
                if (mic) mic.classList.remove("recording");
                const stopBtn = document.getElementById("stop-listening-btn");
                if (stopBtn) stopBtn.style.display = "none";
                updateStatus("No microphone detected. Please check your audio settings.");
                return;
            }
            if (event.error === 'network' && shouldKeepListening) {
                if (!window._networkRetries) window._networkRetries = 0;
                window._networkRetries++;
                if (window._networkRetries <= 2) {
                    setTimeout(() => { if (shouldKeepListening) { try { recognition.start(); } catch (e) { } } }, 200);
                    return;
                }
            }
            if (event.error === 'no-speech') {
                updateStatus("No speech detected. Listening...");
            } else {
                updateStatus("Voice recognition notice: " + event.error);
            }
        };
    }

    let finalTranscript = '';
    let silenceTimer = null;
    const SILENCE_TIMEOUT = 12000;

    shouldKeepListening = true;
    accumulatedTranscript = '';

    recognition.onstart = () => {
        isListening = true;
        updateStatus("Listening for your answer...");
        const mic = document.getElementById("mic-icon");
        if (mic) mic.classList.add("recording");
        const stopBtn = document.getElementById("stop-listening-btn");
        if (stopBtn) stopBtn.style.display = "block";
        const liveEl = document.getElementById("live-transcript");
        if (liveEl) liveEl.textContent = "Listening... Speak clearly into your microphone.";
    };

    recognition.onresult = (event) => {
        if (silenceTimer) clearTimeout(silenceTimer);

        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + ' ';
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        const displayText = accumulatedTranscript + finalTranscript + interimTranscript;
        const liveEl = document.getElementById("live-transcript");
        if (liveEl) liveEl.textContent = displayText || "Listening...";

        silenceTimer = setTimeout(() => {
            shouldKeepListening = false;
            if (recognition && isListening) recognition.stop();
        }, SILENCE_TIMEOUT);
    };

    recognition.onend = () => {
        isListening = false;
        if (silenceTimer) clearTimeout(silenceTimer);
        accumulatedTranscript += finalTranscript;

        const mic = document.getElementById("mic-icon");
        if (mic) mic.classList.remove("recording");

        if (shouldKeepListening) {
            finalTranscript = '';
            setTimeout(() => {
                if (shouldKeepListening) {
                    try { recognition.start(); } catch (e) { }
                }
            }, 100);
        } else {
            const stopBtn = document.getElementById("stop-listening-btn");
            if (stopBtn) stopBtn.style.display = "none";

            if (accumulatedTranscript.trim().length > 0) {
                const answerText = accumulatedTranscript.trim();
                accumulatedTranscript = '';
                const liveEl = document.getElementById("live-transcript");
                if (liveEl) liveEl.textContent = "Answer captured. Evaluating answer & generating feedback...";
                sendUserResponse(answerText);
            } else {
                updateStatus("Awaiting your answer...");
            }
        }
    };

    try {
        recognition.start();
    } catch (e) {
        console.error("Recognition start error:", e);
    }
}

function stopListening() {
    shouldKeepListening = false;
    if (recognition && isListening) {
        recognition.stop();
    }
    const mic = document.getElementById("mic-icon");
    if (mic) mic.classList.remove("recording");
    const stopBtn = document.getElementById("stop-listening-btn");
    if (stopBtn) stopBtn.style.display = "none";
}

// Typed Answer Submission
function submitTextAnswer() {
    const textInput = document.getElementById('text-answer-input');
    const answer = textInput.value.trim();

    if (answer.length === 0) {
        alert("Please write your answer before submitting!");
        return;
    }

    const wasPasted = Boolean(isPasteDetected);

    if (window.speechSynthesis) window.speechSynthesis.cancel();
    stopListening();
    sendUserResponse(answer, wasPasted);
}

// Send user answer and process evaluation
async function sendUserResponse(text, pasteDetected = false) {
    if (!text || !text.trim() || isSubmittingResponse) return;
    isSubmittingResponse = true;

    if (window.speechSynthesis) window.speechSynthesis.cancel();

    const submitBtn = document.getElementById("submit-text-btn");
    const stopBtn = document.getElementById("stop-listening-btn");
    const textInput = document.getElementById("text-answer-input");
    if (submitBtn) submitBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    if (textInput) textInput.disabled = true;

    addMessage("user", text.trim());
    updateStatus("Evaluating answer & generating feedback...");

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: sessionId,
                user_message: text.trim(),
                paste_detected: Boolean(pasteDetected)
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        // Reset paste flag and clear answer input ONLY AFTER the answer has been successfully submitted
        resetPasteDetection();
        if (textInput) {
            textInput.value = '';
        }
        updateWordCount();

        const isComplete = Boolean(data.is_interview_over || data.is_completed || data.action === "close");

        // 1. Render Rubric Evaluation Card if returned (never for closing turns)
        if (data.evaluation && !data.is_interview_over && data.action !== "close" && !isComplete) {
            renderRubricCard(data.evaluation);
            evaluationsHistory.push(data.evaluation);

            // Record score by question number so follow-up answers on the same question
            // update that question's score instead of adding extra marks beyond totalQuestions * 20
            const qNum = data.evaluation.question_number;
            if (qNum !== null && qNum !== undefined && qNum > 0) {
                if (data.evaluation.paste_detected) {
                    if (questionScores[qNum] === undefined) {
                        questionScores[qNum] = 0;
                    }
                } else {
                    const evalScore = data.evaluation.rubric?.total_score || 0;
                    if (questionScores[qNum] === undefined || evalScore >= questionScores[qNum]) {
                        questionScores[qNum] = evalScore;
                    }
                }
            }

            // Update separate current answer score pill
            const currentAnswerEl = document.getElementById("current-answer-score-text");
            if (currentAnswerEl && data.evaluation.rubric) {
                currentAnswerEl.textContent = `${data.evaluation.rubric.total_score} / 20`;
            }

            // Use validated backend session score if provided, else compute from question map
            sessionTotalScore = (data.session_total_score !== undefined && data.session_total_score !== null)
                ? data.session_total_score
                : Object.values(questionScores).reduce((a, b) => a + b, 0);

            answeredQuestionsCount = Object.keys(questionScores).length;

            // Update live running score in topbar using full interview maximum convention
            const runningEl = document.getElementById("running-score-text");
            if (runningEl) {
                const maxScore = (data.max_possible_score !== undefined && data.max_possible_score !== null && data.max_possible_score > 0)
                    ? data.max_possible_score
                    : (totalQuestions * 20);
                runningEl.textContent = `${sessionTotalScore} / ${maxScore}`;
            }
        }

        // 2. Handle interview completion with authoritative state
        if (isComplete) {
            isInterviewCompleted = true;
            currentQuestionIndex = totalQuestions;

            // Immediately stop timer and speech recognition
            stopTimer();
            stopListening();

            const progEl = document.getElementById("q-progress-text");
            if (progEl) {
                progEl.textContent = `Completed (${totalQuestions} of ${totalQuestions})`;
            }

            // Disable answer submission and input controls
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = "<span>Interview Complete ✅</span>";
            }
            if (textInput) {
                textInput.disabled = true;
                textInput.placeholder = "Interview completed. Generating final performance report...";
            }
            if (stopBtn) stopBtn.style.display = "none";
            const mic = document.getElementById("mic-icon");
            if (mic) {
                mic.classList.remove("recording");
                mic.onclick = null;
            }

            const endBtn = document.getElementById("end-btn");
            if (endBtn) {
                endBtn.disabled = true;
                endBtn.innerHTML = "<span>Generating Final Report... ⏳</span>";
            }

            updateStatus("Interview completed. Loading final performance report...");

            // Render closing agent message
            addMessage("agent", data.agent_message);

            // Play voice closing without holding report generation hostage
            speak(data.agent_message);

            // Guaranteed trigger to final report generation
            setTimeout(() => {
                endInterview();
            }, 1000);

            return;
        }

        // 3. Progress tracking
        const progEl = document.getElementById("q-progress-text");
        if (currentQuestionIndex === 0) {
            currentQuestionIndex = 1;
            if (progEl) {
                progEl.textContent = `Question 1 of ${totalQuestions}`;
            }
        } else {
            if (data.action === "advance") {
                if (currentQuestionIndex < totalQuestions) {
                    currentQuestionIndex++;
                }
            }
            if (progEl) {
                progEl.textContent = `Question ${Math.min(currentQuestionIndex, totalQuestions)} of ${totalQuestions}`;
            }
        }

        // 4. Render next interviewer question / follow-up
        addMessage("agent", data.agent_message);
        speak(data.agent_message);

    } catch (err) {
        console.error("Chat turn failed", err);
        updateStatus("Connection Error. Please retry.");
    } finally {
        isSubmittingResponse = false;
        if (!isInterviewCompleted) {
            if (submitBtn) submitBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = false;
            if (textInput) textInput.disabled = false;
        }
    }
}

// Final Interview Completion & Report Generation
async function endInterview() {
    if (isGeneratingReport) return;
    isGeneratingReport = true;
    isInterviewCompleted = true;

    // 1. Immediately stop timer, speech, and listening
    stopTimer();
    stopListening();
    if (synth && synth.speaking) synth.cancel();

    // 2. Disable end-btn to prevent duplicate clicks
    const endBtn = document.getElementById("end-btn");
    if (endBtn) {
        endBtn.disabled = true;
        endBtn.innerHTML = "<span>Generating Final Report... ⏳</span>";
    }

    // 3. Transition UI to report screen
    navigateTo("feedback");

    // 4. Show clear loading state in feedback-content
    const feedbackContent = document.getElementById("feedback-content");
    if (feedbackContent) {
        feedbackContent.innerHTML = `
            <div class="feedback-loading">
                <div class="spinner"></div>
                <p>Synthesizing comprehensive placement feedback and rubric analysis...</p>
            </div>
        `;
    }

    updateStatus("Generating Final Report...");

    try {
        const response = await fetch(`${API_URL}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (HTTP ${response.status})`);
        }

        const data = await response.json();
        renderFeedback(data.feedback);
        updateStatus("Interview Feedback Ready");

        // Save completed session to localStorage history
        saveSessionToHistory({
            sessionId: sessionId,
            date: new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }),
            role: document.getElementById("role-select") ? document.getElementById("role-select").value : "Software Engineer",
            difficulty: document.getElementById("difficulty-select") ? document.getElementById("difficulty-select").value : "Medium",
            track: document.getElementById("type-select") ? document.getElementById("type-select").value : "Technical",
            summary: data.feedback.performance_summary || null,
            verdict: data.feedback.verdict || "Completed"
        });

        // Trigger resume scoring if resume was uploaded
        if (resumeText) {
            scoreResume();
        }
    } catch (err) {
        console.error("Feedback generation failed", err);
        updateStatus("Report Generation Notice");
        if (feedbackContent) {
            feedbackContent.innerHTML = `
                <div class="feedback-section" style="border-color: var(--danger-border); text-align: center; padding: 2rem;">
                    <h4 style="color: var(--danger);">Unable to generate comprehensive report at this moment</h4>
                    <p style="margin: 0.5rem 0 1rem; color: var(--text-muted);">${escapeHtml(err.message)}</p>
                    <button class="btn btn-secondary retry-report-btn" onclick="retryGenerateReport()">
                        <span>Try Loading Report Again 🔄</span>
                    </button>
                </div>
            `;
        }
    } finally {
        isGeneratingReport = false;
    }
}

function retryGenerateReport() {
    isGeneratingReport = false;
    endInterview();
}

// Render Comprehensive Final Performance Report
function renderFeedback(feedback) {
    const summary = feedback.performance_summary;

    if (summary) {
        // Executive Summary Cards
        document.getElementById("rep-q-count").textContent = summary.questions_answered;
        document.getElementById("rep-total-marks").textContent = `${summary.total_marks_obtained} / ${summary.max_possible_marks}`;
        document.getElementById("rep-percentage").textContent = `${summary.overall_percentage}%`;
        document.getElementById("rep-avg-score").textContent = `${summary.average_score} / 20`;

        // Placement Readiness Badge
        const rBadge = document.getElementById("rep-readiness-badge");
        if (summary.overall_percentage >= 80) {
            rBadge.textContent = "Placement Ready ⭐";
            rBadge.className = "readiness-badge badge-ready";
        } else if (summary.overall_percentage >= 60) {
            rBadge.textContent = "Promising Candidate 👍";
            rBadge.className = "readiness-badge badge-promising";
        } else {
            rBadge.textContent = "Needs Practice 📚";
            rBadge.className = "readiness-badge badge-practice";
        }

        // Criterion Progress Bars
        const crits = summary.criterion_averages || {};
        updateCriterionBar("correctness", crits.correctness || 0);
        updateCriterionBar("relevance", crits.relevance || 0);
        updateCriterionBar("clarity", crits.clarity || 0);
        updateCriterionBar("completeness", crits.completeness || 0);

        // Strongest & Weakest Criteria
        document.getElementById("rep-strongest-crit").textContent = summary.strongest_criterion || "N/A";
        document.getElementById("rep-weakest-crit").textContent = summary.weakest_criterion || "N/A";
    }

    // Qualitative Feedback Section
    const container = document.getElementById("feedback-content");
    container.innerHTML = "";

    const cardTitle = document.createElement("div");
    cardTitle.className = "section-card-title";
    cardTitle.innerHTML = `
        <h3>Hiring Assessment &amp; Placement Coaching</h3>
        <p>Expert evaluation based on university placement benchmarks.</p>
    `;
    container.appendChild(cardTitle);

    // Hiring Verdict Box
    const verdictBox = document.createElement("div");
    verdictBox.className = "verdict-box";
    const vClass = "verdict verdict-" + (feedback.verdict || "hire").toLowerCase().replace(/\s+/g, "-");
    verdictBox.innerHTML = `
        <span class="${vClass}">${feedback.verdict}</span>
        <strong style="margin-left: 0.75rem; color: var(--text-main);">Hiring Recommendation</strong>
        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-muted);">${escapeHtml(feedback.verdict_reason)}</p>
    `;
    container.appendChild(verdictBox);

    // Overall Performance
    const perfSec = createFeedbackSection("Executive Summary", feedback.overall_performance);
    container.appendChild(perfSec);

    // Overall Strengths
    const strengthsSec = createFeedbackListSection("Key Demonstrated Strengths", feedback.strengths);
    container.appendChild(strengthsSec);

    // Areas for Improvement
    const improvSec = createFeedbackListSection("Actionable Areas for Improvement", feedback.areas_for_improvement);
    container.appendChild(improvSec);

    // Personalized Preparation Recommendations
    const recs = (summary && summary.personalized_recommendations) ? summary.personalized_recommendations : feedback.key_recommendations;
    const recSec = createFeedbackListSection("Personalized Placement Preparation Recommendations", recs);
    container.appendChild(recSec);
}

function updateCriterionBar(key, avgScore) {
    const textEl = document.getElementById(`score-${key}-text`);
    const barEl = document.getElementById(`score-${key}-bar`);
    if (textEl) textEl.textContent = `${avgScore} / 5`;
    if (barEl) barEl.style.width = `${(avgScore / 5) * 100}%`;
}

function createFeedbackSection(title, content) {
    const sec = document.createElement("div");
    sec.className = "feedback-section";
    sec.style.marginTop = "1.25rem";
    sec.innerHTML = `
        <h4>${title}</h4>
        <p>${escapeHtml(content)}</p>
    `;
    return sec;
}

function createFeedbackListSection(title, items) {
    const sec = document.createElement("div");
    sec.className = "feedback-section";
    sec.style.marginTop = "1.25rem";

    const h = document.createElement("h4");
    h.textContent = title;
    sec.appendChild(h);

    const ul = document.createElement("ul");
    (items || []).forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        ul.appendChild(li);
    });
    sec.appendChild(ul);
    return sec;
}

// Resume ATS Review
async function scoreResume() {
    const container = document.getElementById("resume-score");
    container.classList.remove("hidden");
    container.innerHTML = `
        <div style="text-align: center; padding: 1.5rem;">
            <div class="spinner"></div>
            <p>Analyzing resume against job requirements and cross-checking transcript...</p>
        </div>
    `;

    try {
        const response = await fetch(`${API_URL}/score_resume`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                resume_text: resumeText,
                role: document.getElementById("role-select").value,
                job_description: jobDescription || null,
                session_id: sessionId
            })
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderResumeScore(await response.json());
    } catch (err) {
        console.error("Resume scoring failed", err);
        container.innerHTML = `<p style="color: var(--danger);">Could not complete resume review.</p>`;
    }
}

function renderResumeScore(score) {
    const container = document.getElementById("resume-score");
    container.innerHTML = "";

    const head = document.createElement("div");
    head.className = "section-card-title";
    head.innerHTML = `
        <h3>Resume Review &amp; Claim Verification</h3>
        <p>Deterministic ATS scan and cross-reference of resume claims against spoken answers.</p>
    `;
    container.appendChild(head);

    const scoreTotal = document.createElement("div");
    scoreTotal.style.fontSize = "1.25rem";
    scoreTotal.style.fontWeight = "800";
    scoreTotal.style.marginBottom = "1rem";
    scoreTotal.style.color = "#ffffff";
    scoreTotal.textContent = `Resume Match Score: ${score.total} / 100`;
    container.appendChild(scoreTotal);

    if (score.narrative) {
        const p = document.createElement("p");
        p.style.fontSize = "0.875rem";
        p.style.lineHeight = "1.6";
        p.style.marginBottom = "1rem";
        p.style.color = "var(--text-secondary)";
        p.textContent = score.narrative;
        container.appendChild(p);
    }

    if (score.claim_analysis && score.claim_analysis.checks.length) {
        const h = document.createElement("h4");
        h.style.margin = "1rem 0 0.5rem";
        h.style.color = "#ffffff";
        h.textContent = "Resume Claims vs Interview Demonstration:";
        container.appendChild(h);

        const ul = document.createElement("ul");
        ul.style.listStyle = "none";
        score.claim_analysis.checks.forEach(check => {
            const li = document.createElement("li");
            li.style.marginBottom = "0.5rem";
            li.style.fontSize = "0.8125rem";
            li.style.color = "var(--text-secondary)";
            li.innerHTML = `<strong>[${check.status.toUpperCase()}]</strong> ${escapeHtml(check.claim)} <br><em>&ldquo;${escapeHtml(check.evidence)}&rdquo;</em>`;
            ul.appendChild(li);
        });
        container.appendChild(ul);
    }
}

// ==============================================================================
// LocalStorage Practice History & Analytics Management
// ==============================================================================

const STORAGE_KEY = "interviewace_practice_history";

function getStoredHistory() {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : [];
    } catch (e) {
        return [];
    }
}

function saveSessionToHistory(sessionData) {
    try {
        const history = getStoredHistory();
        history.unshift(sessionData);
        // Keep last 25 sessions
        if (history.length > 25) history.pop();
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
        console.warn("Could not save history to localStorage", e);
    }
}

function renderDashboard() {
    const history = getStoredHistory();

    const totalSessionsEl = document.getElementById("dash-total-sessions");
    const sessionsSubEl = document.getElementById("dash-sessions-sub");
    const avgScoreEl = document.getElementById("dash-avg-score");
    const avgSubEl = document.getElementById("dash-avg-sub");
    const readinessValEl = document.getElementById("dash-readiness-val");
    const readinessSubEl = document.getElementById("dash-readiness-sub");
    const totalQEl = document.getElementById("dash-total-questions");
    const questionsSubEl = document.getElementById("dash-questions-sub");
    const tableContainer = document.getElementById("dash-recent-history-container");

    if (!history || history.length === 0) {
        if (totalSessionsEl) totalSessionsEl.textContent = "0";
        if (sessionsSubEl) sessionsSubEl.textContent = "No completed sessions yet";
        if (avgScoreEl) avgScoreEl.textContent = "--";
        if (avgSubEl) avgSubEl.textContent = "Complete a session to view";
        if (readinessValEl) readinessValEl.textContent = "Needs Practice";
        if (readinessSubEl) readinessSubEl.textContent = "Start your first mock interview";
        if (totalQEl) totalQEl.textContent = "0";
        if (questionsSubEl) questionsSubEl.textContent = "20 marks each question";
        if (tableContainer) {
            tableContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📝</div>
                    <h4>No Practice Sessions Recorded Yet</h4>
                    <p>Start your first mock interview session to record your performance history.</p>
                    <button class="btn btn-primary btn-sm" onclick="navigateTo('setup')" style="margin-top: 1rem;">
                        <span>Start Practice Session</span>
                    </button>
                </div>
            `;
        }
        return;
    }

    // Compute metrics
    const totalSessions = history.length;
    let totalScoreSum = 0;
    let totalQuestionsCount = 0;
    let validScoresCount = 0;

    history.forEach(item => {
        if (item.summary) {
            totalScoreSum += Number(item.summary.overall_percentage) || 0;
            totalQuestionsCount += Number(item.summary.questions_answered) || 0;
            validScoresCount++;
        }
    });

    const avgPct = validScoresCount > 0 ? (totalScoreSum / validScoresCount).toFixed(1) : 0;
    let readinessText = "Needs Practice";
    if (avgPct >= 80) readinessText = "Placement Ready ⭐";
    else if (avgPct >= 60) readinessText = "Promising Candidate 👍";

    if (totalSessionsEl) totalSessionsEl.textContent = totalSessions;
    if (sessionsSubEl) sessionsSubEl.textContent = `${totalSessions} mock session${totalSessions > 1 ? 's' : ''} completed`;
    if (avgScoreEl) avgScoreEl.textContent = `${avgPct}%`;
    if (avgSubEl) avgSubEl.textContent = `Average across ${validScoresCount} session${validScoresCount > 1 ? 's' : ''}`;
    if (readinessValEl) readinessValEl.textContent = readinessText;
    if (readinessSubEl) readinessSubEl.textContent = "Based on comprehensive rubric";
    if (totalQEl) totalQEl.textContent = totalQuestionsCount;
    if (questionsSubEl) questionsSubEl.textContent = `${totalQuestionsCount * 20} max marks evaluated`;

    // Render Recent Sessions table
    if (tableContainer) {
        let rowsHtml = "";
        history.slice(0, 5).forEach((item, idx) => {
            const summary = item.summary || {};
            const pct = summary.overall_percentage || 0;
            const marks = `${summary.total_marks_obtained || 0} / ${summary.max_possible_marks || 0}`;
            const badgeClass = pct >= 80 ? "badge-ready" : pct >= 60 ? "badge-promising" : "badge-practice";
            const badgeLabel = pct >= 80 ? "Ready ⭐" : pct >= 60 ? "Promising 👍" : "Practice 📚";

            rowsHtml += `
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                            <strong style="color: #ffffff; font-size: 0.95rem;">${escapeHtml(item.role)}</strong>
                            <span class="badge badge-outline">${escapeHtml(item.difficulty)}</span>
                        </div>
                        <span style="font-size: 0.75rem; color: var(--text-muted);">${escapeHtml(item.date)} • ${summary.questions_answered || 0} Questions Attempted</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="text-align: right;">
                            <strong style="color: #ffffff; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;">${marks}</strong>
                            <div style="font-size: 0.75rem; color: #a5b4fc;">${pct}%</div>
                        </div>
                        <span class="readiness-badge ${badgeClass}">${badgeLabel}</span>
                    </div>
                </div>
            `;
        });

        tableContainer.innerHTML = rowsHtml;
    }
}

function renderHistoryScreen() {
    const history = getStoredHistory();
    const container = document.getElementById("history-screen-container");
    if (!container) return;

    if (!history || history.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📜</div>
                <h4>No Practice History Yet</h4>
                <p>Complete a mock interview session to automatically log it in your practice portfolio.</p>
                <button class="btn btn-primary btn-sm" onclick="navigateTo('setup')" style="margin-top: 1rem;">
                    <span>Start Practice Session</span>
                </button>
            </div>
        `;
        return;
    }

    let html = `<div style="padding: 0.5rem 0;">`;
    history.forEach((item, idx) => {
        const summary = item.summary || {};
        const pct = summary.overall_percentage || 0;
        const marks = `${summary.total_marks_obtained || 0} / ${summary.max_possible_marks || 0}`;
        const badgeClass = pct >= 80 ? "badge-ready" : pct >= 60 ? "badge-promising" : "badge-practice";
        const badgeLabel = pct >= 80 ? "Placement Ready ⭐" : pct >= 60 ? "Promising 👍" : "Needs Practice 📚";

        html += `
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border);">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem;">
                        <strong style="color: #ffffff; font-size: 1rem;">${escapeHtml(item.role)}</strong>
                        <span class="badge badge-primary">${escapeHtml(item.track || 'Technical')}</span>
                        <span class="badge badge-outline">${escapeHtml(item.difficulty)}</span>
                    </div>
                    <span style="font-size: 0.8125rem; color: var(--text-muted);">${escapeHtml(item.date)} • ${summary.questions_answered || 0} technical questions • Average ${summary.average_score || 0}/20</span>
                </div>
                <div style="display: flex; align-items: center; gap: 1.5rem;">
                    <div style="text-align: right;">
                        <div style="font-size: 1.1rem; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace;">${marks}</div>
                        <div style="font-size: 0.8125rem; color: #a5b4fc;">${pct}%</div>
                    </div>
                    <span class="readiness-badge ${badgeClass}">${badgeLabel}</span>
                </div>
            </div>
        `;
    });
    html += `</div>`;
    container.innerHTML = html;
}

function renderAnalyticsScreen() {
    const history = getStoredHistory();
    const container = document.getElementById("analytics-content-container");
    if (!container) return;

    if (!history || history.length === 0) {
        container.innerHTML = `
            <div class="report-section-card">
                <div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <h4>No Analytical Data Available</h4>
                    <p>Complete mock interviews to generate skill breakdowns across the 4 assessment criteria.</p>
                </div>
            </div>
        `;
        return;
    }

    // Calculate aggregated criteria averages
    let corrSum = 0, relSum = 0, claSum = 0, compSum = 0, count = 0;
    history.forEach(item => {
        if (item.summary && item.summary.criterion_averages) {
            const c = item.summary.criterion_averages;
            corrSum += c.correctness || 0;
            relSum += c.relevance || 0;
            claSum += c.clarity || 0;
            compSum += c.completeness || 0;
            count++;
        }
    });

    const corrAvg = count > 0 ? (corrSum / count).toFixed(1) : 0;
    const relAvg = count > 0 ? (relSum / count).toFixed(1) : 0;
    const claAvg = count > 0 ? (claSum / count).toFixed(1) : 0;
    const compAvg = count > 0 ? (compSum / count).toFixed(1) : 0;

    container.innerHTML = `
        <div class="report-section-card">
            <div class="section-card-title">
                <h3>Overall Criteria Performance Across ${count} Completed Session${count > 1 ? 's' : ''}</h3>
                <p>Aggregated benchmark data across all practiced technical interviews.</p>
            </div>
            <div class="criteria-analytics-grid">
                <div class="criterion-bar-card">
                    <div class="bar-header">
                        <span class="bar-title">Correctness &amp; Technical Accuracy</span>
                        <strong>${corrAvg} / 5</strong>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${(corrAvg / 5) * 100}%;"></div>
                    </div>
                    <span class="bar-hint">Factual correctness and algorithm mechanics</span>
                </div>
                <div class="criterion-bar-card">
                    <div class="bar-header">
                        <span class="bar-title">Relevance &amp; Direct Response</span>
                        <strong>${relAvg} / 5</strong>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${(relAvg / 5) * 100}%;"></div>
                    </div>
                    <span class="bar-hint">Answering on-topic without deflection</span>
                </div>
                <div class="criterion-bar-card">
                    <div class="bar-header">
                        <span class="bar-title">Clarity &amp; Communication Structure</span>
                        <strong>${claAvg} / 5</strong>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${(claAvg / 5) * 100}%;"></div>
                    </div>
                    <span class="bar-hint">Logical articulation and structured explanation</span>
                </div>
                <div class="criterion-bar-card">
                    <div class="bar-header">
                        <span class="bar-title">Completeness &amp; Edge Cases</span>
                        <strong>${compAvg} / 5</strong>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${(compAvg / 5) * 100}%;"></div>
                    </div>
                    <span class="bar-hint">Handling trade-offs, constraints, and edge cases</span>
                </div>
            </div>
        </div>
    `;
}
