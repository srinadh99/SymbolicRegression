(function () {
  "use strict";

  var slider = document.getElementById("z-slider");
  var number = document.getElementById("z-number");
  var select = document.getElementById("experiment");
  var cards = document.getElementById("cards");
  var hint = document.getElementById("experiment-hint");
  var errorBox = document.getElementById("error");

  var zMin = parseFloat(slider.min);
  var zMax = parseFloat(slider.max);

  var requestId = 0;
  var timer = null;

  function clamp(value) {
    if (isNaN(value)) {
      return parseFloat(slider.value);
    }
    return Math.min(zMax, Math.max(zMin, value));
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = !message;
  }

  function buildCard(prediction) {
    var card = document.createElement("article");
    card.className = "card";

    var model = document.createElement("div");
    model.className = "card-model";
    model.textContent = "Model: " + prediction.model;
    card.appendChild(model);

    var label = document.createElement("div");
    label.className = "card-label";
    label.textContent = "Predicted Class";
    card.appendChild(label);

    var value = document.createElement("div");
    value.className = "card-class";
    if (prediction.predicted_class) {
      value.textContent = prediction.predicted_class;
      value.dataset.class = prediction.predicted_class;
    } else {
      card.classList.add("is-error");
      value.textContent = prediction.error || "unavailable";
    }
    card.appendChild(value);

    if (prediction.expression) {
      card.appendChild(buildExpressionBlock(prediction));
    }
    return card;
  }

  function buildExpressionBlock(prediction) {
    var block = document.createElement("div");
    block.className = "card-extra";

    var caption = document.createElement("span");
    caption.textContent = "Expression";
    block.appendChild(caption);

    var code = document.createElement("code");
    code.textContent = prediction.expression;
    block.appendChild(code);

    if (prediction.thresholds) {
      var parts = [];
      if (typeof prediction.score === "number") {
        parts.push("s(z) = " + prediction.score.toFixed(4));
      }
      parts.push("t₁ = " + prediction.thresholds.t1.toFixed(4));
      parts.push("t₂ = " + prediction.thresholds.t2.toFixed(4));

      var numbers = document.createElement("span");
      numbers.className = "numbers";
      numbers.textContent = parts.join("  ·  ");
      block.appendChild(numbers);
    }
    return block;
  }

  function render(payload) {
    cards.textContent = "";
    payload.predictions.forEach(function (prediction) {
      cards.appendChild(buildCard(prediction));
    });
    hint.textContent =
      payload.experiment.catalogue +
      " · classes: " +
      payload.experiment.classes.join(", ");
  }

  function refresh() {
    var z = clamp(parseFloat(slider.value));
    var id = ++requestId;
    var url =
      "/api/predict?z=" + encodeURIComponent(z) +
      "&experiment=" + encodeURIComponent(select.value);

    cards.classList.add("is-stale");

    fetch(url)
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (body) {
            throw new Error(body.detail || "Request failed (" + response.status + ")");
          });
        }
        return response.json();
      })
      .then(function (payload) {
        // Ignore replies that arrived after a newer request went out.
        if (id !== requestId) {
          return;
        }
        showError("");
        render(payload);
      })
      .catch(function (err) {
        if (id === requestId) {
          showError(err.message);
        }
      })
      .finally(function () {
        if (id === requestId) {
          cards.classList.remove("is-stale");
        }
      });
  }

  function scheduleRefresh() {
    window.clearTimeout(timer);
    timer = window.setTimeout(refresh, 90);
  }

  function syncFromSlider() {
    number.value = parseFloat(slider.value).toFixed(2);
    scheduleRefresh();
  }

  function syncFromNumber() {
    var z = clamp(parseFloat(number.value));
    slider.value = z;
    number.value = z.toFixed(2);
    scheduleRefresh();
  }

  slider.addEventListener("input", syncFromSlider);
  number.addEventListener("change", syncFromNumber);
  select.addEventListener("change", refresh);

  syncFromSlider();
})();
