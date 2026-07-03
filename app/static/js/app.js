document.addEventListener("DOMContentLoaded", () => {
  CanvasController.init("drawCanvas", "brushSize");

  const clearBtn = document.getElementById("clearBtn");
  const predictBtn = document.getElementById("predictBtn");
  const predictionChar = document.getElementById("predictionChar");
  const confidenceBar = document.getElementById("confidenceBar");
  const confidenceLabel = document.getElementById("confidenceLabel");
  const topkList = document.getElementById("topkList");
  const statusDot = document.querySelector(".status-dot");
  const statusText = document.getElementById("statusText");

  checkHealth();

  clearBtn.addEventListener("click", () => {
    CanvasController.resetCanvas();
    predictionChar.textContent = "?";
    confidenceBar.style.width = "0%";
    confidenceLabel.textContent = "AWAITING INPUT";
    topkList.innerHTML = "";
  });

  predictBtn.addEventListener("click", () => {
    if (CanvasController.isEmpty()) {
      confidenceLabel.textContent = "DRAW SOMETHING FIRST";
      return;
    }
    predictBtn.disabled = true;
    predictBtn.textContent = "ANALYZING...";

    CanvasController.toBlob((blob) => {
      const formData = new FormData();
      formData.append("image", blob, "input.png");

      fetch("/api/predict", { method: "POST", body: formData })
        .then((res) => res.json())
        .then((data) => {
          predictBtn.disabled = false;
          predictBtn.textContent = "ANALYZE";
          if (data.error) {
            confidenceLabel.textContent = data.error.toUpperCase();
            return;
          }
          renderResult(data);
        })
        .catch((err) => {
          predictBtn.disabled = false;
          predictBtn.textContent = "ANALYZE";
          confidenceLabel.textContent = "REQUEST FAILED";
          console.error(err);
        });
    });
  });

  function renderResult(data) {
    predictionChar.textContent = data.prediction;
    const pct = Math.round(data.confidence * 100);
    confidenceBar.style.width = pct + "%";
    confidenceLabel.textContent = `CONFIDENCE: ${pct}%  ${data.is_confident ? "// HIGH CERTAINTY" : "// LOW CERTAINTY"}`;

    topkList.innerHTML = "";
    (data.top_k || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = "topk-item";
      row.innerHTML = `<span class="char">${item.character}</span><span class="pct">${Math.round(item.confidence * 100)}%</span>`;
      topkList.appendChild(row);
    });
  }

  function checkHealth() {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => {
        if (data.model_ready) {
          statusDot.classList.add("online");
          statusText.textContent = "MODEL ONLINE";
        } else {
          statusDot.classList.add("offline");
          statusText.textContent = "MODEL NOT TRAINED";
        }
      })
      .catch(() => {
        statusDot.classList.add("offline");
        statusText.textContent = "API UNREACHABLE";
      });
  }
});
