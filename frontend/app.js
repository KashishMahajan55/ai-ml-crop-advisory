/**
 * frontend/app.js
 * ---------------
 * Synthetic dataset generated for educational AI/ML demonstration.
 *
 * Connects the HTML form to the FastAPI backend.
 * Replace BACKEND_URL with your Render deployment URL after deploying.
 */

// ── CONFIG ───────────────────────────────────────────────────────────────────
// Empty string = same server (works locally AND on Render — no CORS ever)
// After deploying backend to Render, replace "" with your Render URL
const BACKEND_URL = "";

const CROP_EMOJI = {
  Rice: "🌾", Wheat: "🌾", Maize: "🌽", Chickpea: "🫘",
  Lentil: "🫘", Cotton: "🌿", Sugarcane: "🎋", Soybean: "🫘",
  Groundnut: "🥜", Mungbean: "🫘", Blackgram: "🫘",
  Pigeonpea: "🫘", Jute: "🌿", Coffee: "☕", Apple: "🍎",
  Mango: "🥭", Grapes: "🍇", Watermelon: "🍉",
  Papaya: "🍈", Coconut: "🥥",
};

const MEDALS = ["🥇", "🥈", "🥉"];

// ── slider value display ──────────────────────────────────────────────────────
function updateVal(id) {
  document.getElementById(`${id}-val`).textContent =
    document.getElementById(id).value;
}

// ── form submit ───────────────────────────────────────────────────────────────
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const btn = document.getElementById("predict-btn");
  const resultBox = document.getElementById("result-box");
  const errorBox  = document.getElementById("error-box");
  const errorMsg  = document.getElementById("error-msg");

  // hide previous results
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  btn.disabled  = true;
  btn.textContent = "⏳ Predicting...";

  const payload = {
    nitrogen_N:    parseFloat(document.getElementById("nitrogen_N").value),
    phosphorus_P:  parseFloat(document.getElementById("phosphorus_P").value),
    potassium_K:   parseFloat(document.getElementById("potassium_K").value),
    temperature_C: parseFloat(document.getElementById("temperature_C").value),
    humidity_pct:  parseFloat(document.getElementById("humidity_pct").value),
    soil_pH:       parseFloat(document.getElementById("soil_pH").value),
    rainfall_mm:   parseFloat(document.getElementById("rainfall_mm").value),
  };

  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    console.log("API response:", data);   // visible in browser DevTools → Console
    showResults(data);

  } catch (err) {
    console.error("Predict error:", err);
    errorMsg.textContent = `Error: ${err.message}. Make sure the backend is running.`;
    errorBox.classList.remove("hidden");
  } finally {
    btn.disabled    = false;
    btn.textContent = "🌱 Predict Best Crop";
  }
});

// ── render results ────────────────────────────────────────────────────────────
function showResults(data) {
  const top3Grid  = document.getElementById("top3-cards");
  const yieldVal  = document.getElementById("yield-value");
  const resultBox = document.getElementById("result-box");

  top3Grid.innerHTML = "";

  data.top3.forEach((item, i) => {
    const emoji = CROP_EMOJI[item.crop] || "🌿";
    const card  = document.createElement("div");
    card.className = `crop-card rank-${item.rank}`;
    card.innerHTML = `
      <div class="crop-medal">${MEDALS[i]}</div>
      <div class="crop-name">${emoji} ${item.crop}</div>
      <div class="crop-prob">${item.confidence_pct}% confidence</div>
      <div class="prob-bar-wrap">
        <div class="prob-bar" style="width:${item.confidence_pct}%"></div>
      </div>
    `;
    top3Grid.appendChild(card);
  });

  yieldVal.textContent = `${data.estimated_yield} t/ha`;
  resultBox.classList.remove("hidden");

  // scroll the result into view after a short delay so the DOM has painted
  setTimeout(() => {
    resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

// ── dataset stats ─────────────────────────────────────────────────────────────
async function loadStats() {
  const btn      = document.getElementById("load-stats-btn");
  const statsBox = document.getElementById("stats-box");

  btn.disabled    = true;
  btn.textContent = "⏳ Loading...";
  statsBox.classList.add("hidden");
  statsBox.innerHTML = "";

  try {
    const res = await fetch(`${BACKEND_URL}/dataset/stats`);
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    // total records card
    const totalCard = document.createElement("div");
    totalCard.className = "stat-card";
    totalCard.innerHTML = `
      <h4>📦 Dataset</h4>
      <div class="stat-row"><span>Total records</span><span>${data.total_records.toLocaleString()}</span></div>
      <div class="stat-row"><span>Crop types</span><span>${Object.keys(data.crop_counts).length}</span></div>
      <div class="stat-row"><span>Type</span><span>Synthetic</span></div>
    `;
    statsBox.appendChild(totalCard);

    // feature stats cards
    const featureLabels = {
      nitrogen_N: "Nitrogen (N)", phosphorus_P: "Phosphorus (P)",
      potassium_K: "Potassium (K)", temperature_C: "Temperature",
      humidity_pct: "Humidity", soil_pH: "Soil pH",
      rainfall_mm: "Rainfall", estimated_yield: "Est. Yield",
    };
    for (const [col, stats] of Object.entries(data.feature_stats)) {
      const card = document.createElement("div");
      card.className = "stat-card";
      card.innerHTML = `
        <h4>${featureLabels[col] || col}</h4>
        <div class="stat-row"><span>Min</span><span>${stats.min}</span></div>
        <div class="stat-row"><span>Mean</span><span>${stats.mean}</span></div>
        <div class="stat-row"><span>Max</span><span>${stats.max}</span></div>
      `;
      statsBox.appendChild(card);
    }

    statsBox.classList.remove("hidden");

  } catch (err) {
    const errorDiv = document.createElement("p");
    errorDiv.style.color = "#c53030";
    errorDiv.textContent = `Could not load stats: ${err.message}`;
    statsBox.appendChild(errorDiv);
    statsBox.classList.remove("hidden");
  } finally {
    btn.disabled    = false;
    btn.textContent = "Load Dataset Stats";
  }
}
