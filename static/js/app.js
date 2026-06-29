const form = document.getElementById("predictForm");
const input = document.getElementById("imageInput");
const dropzone = document.getElementById("dropzone");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const resultBox = document.getElementById("resultBox");
const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");

let selectedDataUrl = "";

function setResult(title, detail, isError = false) {
    predictionText.textContent = title;
    confidenceText.textContent = detail;
    resultBox.classList.toggle("is-error", isError);
}

function loadFile(file) {
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        selectedDataUrl = reader.result;
        previewImage.src = selectedDataUrl;
        previewWrap.classList.add("is-visible");
        setResult("Ready", "Run prediction to classify this scan.");
    };
    reader.readAsDataURL(file);
}

input.addEventListener("change", () => loadFile(input.files[0]));

dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("is-dragover");
});

dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
    const file = event.dataTransfer.files[0];
    input.files = event.dataTransfer.files;
    loadFile(file);
});

clearButton.addEventListener("click", () => {
    input.value = "";
    selectedDataUrl = "";
    previewImage.removeAttribute("src");
    previewWrap.classList.remove("is-visible");
    setResult("Waiting for image", "Confidence will appear after prediction.");
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!selectedDataUrl) {
        setResult("No image selected", "Choose a CT scan before prediction.", true);
        return;
    }

    predictButton.disabled = true;
    setResult("Analyzing", "The model is processing the CT image.");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: selectedDataUrl }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        const result = Array.isArray(data) ? data[0] : data;
        const confidence = result.confidence ? `${(result.confidence * 100).toFixed(2)}% confidence` : "Confidence unavailable";
        setResult(result.image, confidence);
    } catch (error) {
        setResult("Prediction unavailable", error.message, true);
    } finally {
        predictButton.disabled = false;
    }
});

