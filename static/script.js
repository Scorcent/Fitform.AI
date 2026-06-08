/**
 * script.js — FITFORM.AI Frontend Logic
 *
 * Handles video upload, result display, chat, language switching,
 * light/dark theme, and the Analyze / Coach Chat view switch.
 */

// ══════════════════════════════════════════════
// Translations
// ══════════════════════════════════════════════
const T = {
    en: {
        // Top bar / tabs
        lang_en: "English",
        lang_es: "Spanish",
        tab_analyze: "Analyze",
        tab_coach: "Coach Chat",
        logout: "Log Out",
        // Analyzer page
        analyze_title: "Analyze Your Form",
        analyze_subtitle: "Choose an exercise, upload your video, and get instant feedback.",
        ex_pushup: "Push-Up",
        ex_squat: "Squat",
        ex_plank: "Plank",
        ex_lunge: "Lunge",
        file_prompt: "Click to select a video file",
        analyze_btn: "Analyze My Form",
        loading: "Analyzing your form...",
        strengths_title: "What you did well",
        tips_title: "How to improve",
        coach_title: "Coach's Notes",
        chat_title: "Ask Your AI Coach",
        chat_subtitle: "Ask anything about your form, technique, or training.",
        chat_placeholder: "e.g. How can I build up to a deeper push-up?",
        send: "Send",
        // Rating labels
        rating_great: "Great Form!",
        rating_okay: "Almost There!",
        rating_needs_work: "Keep Practicing!",
        rating_unknown: "Could Not Analyze",
        // Errors
        err_no_file: "Please select a video file first.",
        err_generic: "Something went wrong.",
        err_server: "Could not connect to the server. Is it running?",
        err_chat: "Could not reach the server.",
        no_feedback: "Upload a video to get personalized coaching feedback.",
        no_response: "No response.",
    },
    es: {
        lang_en: "Ingles",
        lang_es: "Espanol",
        tab_analyze: "Analizar",
        tab_coach: "Chat con Entrenador",
        logout: "Cerrar Sesion",
        analyze_title: "Analiza Tu Forma",
        analyze_subtitle: "Elige un ejercicio, sube tu video y recibe comentarios al instante.",
        ex_pushup: "Flexion",
        ex_squat: "Sentadilla",
        ex_plank: "Plancha",
        ex_lunge: "Zancada",
        file_prompt: "Haz clic para seleccionar un video",
        analyze_btn: "Analizar Mi Forma",
        loading: "Analizando tu forma...",
        strengths_title: "Lo que hiciste bien",
        tips_title: "Como mejorar",
        coach_title: "Notas del Entrenador",
        chat_title: "Pregunta a Tu Entrenador IA",
        chat_subtitle: "Pregunta lo que quieras sobre tu forma, tecnica o entrenamiento.",
        chat_placeholder: "Ej. Como puedo mejorar la profundidad de mis flexiones?",
        send: "Enviar",
        rating_great: "Excelente Forma!",
        rating_okay: "Casi Perfecto!",
        rating_needs_work: "Sigue Practicando!",
        rating_unknown: "No Se Pudo Analizar",
        err_no_file: "Por favor selecciona un archivo de video primero.",
        err_generic: "Algo salio mal.",
        err_server: "No se pudo conectar al servidor. Esta ejecutandose?",
        err_chat: "No se pudo conectar al servidor.",
        no_feedback: "Sube un video para recibir comentarios personalizados.",
        no_response: "Sin respuesta.",
    }
};

// Current language
let currentLang = localStorage.getItem("fitform_lang") || "en";

function setLang(lang) {
    currentLang = lang;
    localStorage.setItem("fitform_lang", lang);

    document.querySelectorAll(".lang-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(`.lang-btn[onclick="setLang('${lang}')"]`).forEach(b => b.classList.add("active"));

    const tr = T[lang];
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.getAttribute("data-i18n");
        if (tr[key]) el.textContent = tr[key];
    });

    document.querySelectorAll("[data-i18n-ph]").forEach(el => {
        const key = el.getAttribute("data-i18n-ph");
        if (tr[key]) el.placeholder = tr[key];
    });

    // Keep the file label in sync if no file is selected yet
    if (fileInput && fileInput.files.length === 0) {
        fileLabel.textContent = tr.file_prompt;
    }
}

function t(key) {
    return T[currentLang][key] || T["en"][key] || key;
}

// ══════════════════════════════════════════════
// Theme (light / dark)
// ══════════════════════════════════════════════
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("fitform_theme", theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(current === "light" ? "dark" : "light");
}

// ══════════════════════════════════════════════
// View switch (Analyze / Coach Chat)
// ══════════════════════════════════════════════
function showView(view) {
    document.getElementById("view-analyze").classList.toggle("hidden", view !== "analyze");
    document.getElementById("view-coach").classList.toggle("hidden", view !== "coach");
    document.querySelectorAll(".view-tab").forEach(b => {
        b.classList.toggle("active", b.getAttribute("data-view") === view);
    });
}

document.querySelectorAll(".view-tab").forEach(btn => {
    btn.addEventListener("click", () => showView(btn.getAttribute("data-view")));
});

// ══════════════════════════════════════════════
// DOM elements
// ══════════════════════════════════════════════
const form         = document.getElementById("upload-form");
const fileInput    = document.getElementById("video-input");
const fileLabel    = document.getElementById("file-name");
const fileLabelBox = document.getElementById("file-label");
const uploadBtn    = document.getElementById("upload-btn");
const loadingDiv   = document.getElementById("loading");
const errorDiv     = document.getElementById("error");
const resultsDiv   = document.getElementById("results");
const chatForm     = document.getElementById("chat-form");
const chatInput    = document.getElementById("chat-input");
const chatBtn      = document.getElementById("chat-btn");
const chatMessages = document.getElementById("chat-messages");

let analysisContext = {};

// ── File selection: enable the button only when a real file is chosen ──
fileInput.addEventListener("change", () => {
    const hasFile = fileInput.files.length > 0;
    if (hasFile) {
        fileLabel.textContent = fileInput.files[0].name;
        fileLabelBox.classList.add("has-file");
    } else {
        fileLabel.textContent = t("file_prompt");
        fileLabelBox.classList.remove("has-file");
    }
    uploadBtn.disabled = !hasFile;
});

// ── Upload handler ──
form.addEventListener("submit", async (event) => {
    event.preventDefault();

    // Hard guard: never analyze without a real video file
    if (fileInput.files.length === 0) {
        showError(t("err_no_file"));
        uploadBtn.disabled = true;
        return;
    }

    const exercise = document.querySelector('input[name="exercise"]:checked').value;

    const formData = new FormData();
    formData.append("video", fileInput.files[0]);
    formData.append("exercise", exercise);
    formData.append("language", currentLang);

    loadingDiv.classList.remove("hidden");
    resultsDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");
    uploadBtn.disabled = true;
    chatMessages.innerHTML = "";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || t("err_generic"));
            return;
        }

        displayResults(data);

    } catch (err) {
        showError(t("err_server"));
    } finally {
        loadingDiv.classList.add("hidden");
        uploadBtn.disabled = fileInput.files.length === 0;
    }
});

// ── Display results ──
function displayResults(data) {
    analysisContext = {
        exercise: data.exercise,
        overall: data.overall,
        strengths: data.strengths || [],
        tips: data.tips || [],
    };

    const banner  = document.getElementById("rating-banner");
    const label   = document.getElementById("rating-label");
    const summary = document.getElementById("rating-summary");

    banner.className = "rating-banner";

    if (data.overall === "great") {
        banner.classList.add("great");
        label.textContent = t("rating_great");
    } else if (data.overall === "okay") {
        banner.classList.add("okay");
        label.textContent = t("rating_okay");
    } else if (data.overall === "needs work") {
        banner.classList.add("needs-work");
        label.textContent = t("rating_needs_work");
    } else {
        banner.classList.add("needs-work");
        label.textContent = t("rating_unknown");
    }

    summary.textContent = data.summary || "";

    // Strengths
    const strengthsSection = document.getElementById("strengths-section");
    const strengthsList = document.getElementById("strengths-list");
    strengthsList.innerHTML = "";

    if (data.strengths && data.strengths.length > 0) {
        data.strengths.forEach(s => {
            const li = document.createElement("li");
            li.textContent = s;
            strengthsList.appendChild(li);
        });
        strengthsSection.classList.remove("hidden");
    } else {
        strengthsSection.classList.add("hidden");
    }

    // Tips
    const tipsSection = document.getElementById("tips-section");
    const tipsList = document.getElementById("tips-list");
    tipsList.innerHTML = "";

    if (data.tips && data.tips.length > 0) {
        data.tips.forEach(tip => {
            const li = document.createElement("li");
            li.textContent = tip;
            tipsList.appendChild(li);
        });
        tipsSection.classList.remove("hidden");
    } else {
        tipsSection.classList.add("hidden");
    }

    document.getElementById("result-feedback").textContent =
        data.feedback || t("no_feedback");

    resultsDiv.classList.remove("hidden");
    resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove("hidden");
}

// ── Chat ──
chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = chatInput.value.trim();
    if (!message) return;

    addChatBubble(message, "user");
    chatInput.value = "";
    chatBtn.disabled = true;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                context: analysisContext,
                language: currentLang,
            }),
        });

        const data = await response.json();
        addChatBubble(data.reply || t("no_response"), "bot");

    } catch (err) {
        addChatBubble(t("err_chat"), "bot");
    } finally {
        chatBtn.disabled = false;
        chatInput.focus();
    }
});

function addChatBubble(text, sender) {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble chat-" + sender;
    bubble.textContent = text;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ══════════════════════════════════════════════
// "Pop" effect on interactive elements
// ══════════════════════════════════════════════
document.addEventListener("click", (e) => {
    const el = e.target.closest(".btn-primary, .view-tab, .lang-btn, .icon-btn, .exercise-card");
    if (!el) return;
    el.classList.remove("is-popping");
    void el.offsetWidth;  // force reflow to restart the animation
    el.classList.add("is-popping");
});

// ── Init on page load ──
setLang(currentLang);
