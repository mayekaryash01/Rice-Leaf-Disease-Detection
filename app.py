import warnings
warnings.filterwarnings("ignore")

import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="Rice Leaf Disease Detection")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and encoder at startup
checkpoint = torch.load("efficientnetb3model3.pth", map_location=device)
model = models.efficientnet_b3(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

label_encoder = joblib.load("label_encoder3.pkl")

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

disease_details = {
    'bacterial_leaf_blight': {
        'name': 'Bacterial Leaf Blight',
        'status_type': 'error',
        'immediate': """
        <ul>
            <li><strong>Fertilizer Management:</strong> Immediately halt all urea and high-nitrogen fertilizer applications to prevent disease progression.</li>
            <li><strong>Water Drainage:</strong> Drain standing water from the infected field and allow the soil to aerate for 7–10 days.</li>
        </ul>
        """,
        'chemical': """
        <ul>
            <li><strong>Recommended Treatment:</strong> Apply copper-based bactericides such as <strong>Copper Oxychloride (2g/L)</strong> or <strong>Copper Hydroxide</strong> dissolved in clean water.</li>
            <li><strong>Application Method:</strong> Spray foliage thoroughly during the late afternoon.</li>
        </ul>
        """
    },
    'brown_spot': {
        'name': 'Brown Spot Disease',
        'status_type': 'warning',
        'immediate': """
        <ul>
            <li><strong>Immediate Action:</strong> Ensure balanced soil fertilization, especially adequate potassium (potash) application.</li>
            <li><strong>Irrigation Management:</strong> Maintain adequate water depth and prevent the soil from drying out completely.</li>
        </ul>
        """,
        'chemical': """
        <ul>
            <li><strong>Fungicide Treatment:</strong> Spray broad-spectrum fungicides such as <strong>Propiconazole (Tilt 250 EC)</strong> at 1 ml/L or <strong>Carbendazim</strong> at 1–2 g/L. Spray twice at intervals of 10–12 days if symptoms persist.</li>
        </ul>
        """
    },
    'tungro': {
        'name': 'Tungro Virus Disease',
        'status_type': 'error',
        'immediate': """
        <ul>
            <li><strong>Immediate Action:</strong> Tungro is spread by the green leafhopper insect vector. Rapid insect control is critical to stop viral transmission.</li>
            <li><strong>Crop Sanitation:</strong> Uproot and safely dispose of severely infected, stunted plants away from the field.</li>
        </ul>
        """,
        'chemical': """
        <ul>
            <li><strong>Insecticide Treatment:</strong> Apply approved systemic insecticides such as <strong>Imidacloprid</strong>, <strong>Thiamethoxam</strong>, or <strong>Buprofezin</strong> to eliminate vector leafhopper populations.</li>
        </ul>
        """
    },
    'normal': {
        'name': 'Healthy Rice Leaf (No Disease Detected)',
        'status_type': 'success',
        'immediate': """
        <ul>
            <li><strong>Advice:</strong> The plant appears healthy and vigorous. Maintain regular crop monitoring and balanced fertilization.</li>
        </ul>
        """,
        'chemical': """
        <ul>
            <li><strong>Advice:</strong> The crop is healthy; no chemical fungicides or pesticides are required.</li>
        </ul>
        """
    }
}

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Leaf Disease Detection</title>
    <link rel="icon" href="data:,">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        body {
            background-color: #0e1117;
            color: #f9fafb;
            padding: 1rem 2.5rem 3rem 2.5rem;
            min-height: 100vh;
        }
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f9fafb;
            margin-bottom: 1.5rem;
            text-align: left;
        }
        .layout-grid {
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2.5rem;
            align-items: start;
        }
        @media (max-width: 900px) {
            .layout-grid {
                grid-template-columns: 1fr;
            }
            body {
                padding: 1rem 1.25rem;
            }
        }
        .col-card {
            background-color: transparent;
        }
        .upload-label {
            display: block;
            font-size: 0.95rem;
            color: #e5e7eb;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        .dropzone {
            border: 1px dashed rgba(255, 255, 255, 0.25);
            border-radius: 8px;
            padding: 2.25rem 1.5rem;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.02);
            cursor: pointer;
            transition: border-color 0.15s, background-color 0.15s;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: rgba(255, 255, 255, 0.5);
            background-color: rgba(255, 255, 255, 0.04);
        }
        .dropzone-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }
        .dropzone-subtext {
            color: rgba(255, 255, 255, 0.45);
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }
        .browse-btn {
            display: inline-block;
            background-color: #21262d;
            color: #e5e7eb;
            border: 1px solid #30363d;
            padding: 0.45rem 1rem;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 500;
            pointer-events: none;
        }
        #fileInput {
            display: none;
        }
        .preview-box {
            margin-top: 1rem;
            display: none;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #30363d;
            background: #161b22;
        }
        .preview-box img {
            width: 100%;
            max-height: 380px;
            object-fit: contain;
            display: block;
        }
        .preview-caption {
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
            color: #8b949e;
            text-align: center;
            background: #161b22;
        }
        .diagnose-btn {
            width: 100%;
            margin-top: 1rem;
            padding: 0.65rem 1.25rem;
            background-color: #238636;
            color: #ffffff;
            border: 1px solid rgba(240, 246, 252, 0.1);
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.15s;
        }
        .diagnose-btn:hover {
            background-color: #2ea043;
        }
        .diagnose-btn:disabled {
            background-color: #21262d;
            color: #6e7681;
            cursor: not-allowed;
            border-color: #30363d;
        }
        .error-banner {
            margin-top: 0.75rem;
            padding: 0.75rem 1rem;
            background-color: #3b1e1e;
            border: 1px solid #7f1d1d;
            color: #f87171;
            border-radius: 6px;
            font-size: 0.9rem;
            display: none;
        }
        .res-container {
            display: none;
        }
        .res-title {
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 0.85rem;
            color: #f9fafb;
        }
        .status-box {
            padding: 0.85rem 1.15rem;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
        }
        .status-error {
            background-color: #3b1e1e;
            border: 1px solid #7f1d1d;
            color: #f87171;
        }
        .status-warning {
            background-color: #3b2f1e;
            border: 1px solid #78350f;
            color: #fbbf24;
        }
        .status-success {
            background-color: #1e3b2b;
            border: 1px solid #14532d;
            color: #4ade80;
        }
        .tabs-header {
            display: flex;
            border-bottom: 1px solid #30363d;
            margin-bottom: 1rem;
            gap: 0.5rem;
        }
        .tab-btn {
            background: none;
            border: none;
            color: #8b949e;
            padding: 0.6rem 1rem;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: color 0.15s, border-color 0.15s;
        }
        .tab-btn.active {
            color: #f0f6fc;
            border-bottom: 2px solid #f87171;
        }
        .tab-content {
            display: none;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #e5e7eb;
            margin-bottom: 1.5rem;
        }
        .tab-content.active {
            display: block;
        }
        .tab-content ul {
            padding-left: 1.25rem;
        }
        .tab-content li {
            margin-bottom: 0.5rem;
        }
        .expander {
            border: 1px solid #30363d;
            border-radius: 6px;
            background-color: #161b22;
            overflow: hidden;
        }
        .expander-header {
            padding: 0.75rem 1rem;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
            font-weight: 500;
            color: #c9d1d9;
            user-select: none;
        }
        .expander-header:hover {
            background-color: #1c2128;
        }
        .expander-content {
            display: none;
            padding: 0.85rem 1rem;
            border-top: 1px solid #30363d;
            font-size: 0.88rem;
        }
        .prob-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.45rem;
            color: #e5e7eb;
        }
        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1 class="main-title">Rice Leaf Disease Detection</h1>

    <div class="layout-grid">
        <div class="col-card">
            <label class="upload-label">Upload a rice leaf photo</label>
            <div class="dropzone" id="dropzone">
                <span class="browse-btn">Browse files</span>
                <div class="dropzone-text">Drag and drop file here</div>
                <div class="dropzone-subtext">Limit 200MB per file &bull; JPG, JPEG, PNG</div>
            </div>
            <input type="file" id="fileInput" accept="image/jpeg, image/jpg, image/png">

            <div class="preview-box" id="previewBox">
                <img id="previewImg" src="" alt="Uploaded Leaf Preview">
                <div class="preview-caption">Uploaded Image</div>
            </div>

            <button class="diagnose-btn" id="diagnoseBtn">Diagnose Disease</button>
            <div class="error-banner" id="errorBanner"></div>
        </div>

        <div class="col-card">
            <div class="res-container" id="resContainer">
                <div class="res-title">Diagnostic Result:</div>
                <div class="status-box" id="statusBox"></div>

                <div class="tabs-header">
                    <button class="tab-btn active" id="tabBtn1" onclick="switchTab(1)">Immediate Actions</button>
                    <button class="tab-btn" id="tabBtn2" onclick="switchTab(2)">Chemical & Treatment Guide</button>
                </div>

                <div class="tab-content active" id="tabContent1"></div>
                <div class="tab-content" id="tabContent2"></div>

                <div class="expander">
                    <div class="expander-header" onclick="toggleExpander()">
                        <span>Technical Model Diagnostics (Confidence Scores)</span>
                        <span id="expanderArrow">&#9662;</span>
                    </div>
                    <div class="expander-content" id="expanderContent">
                        <div id="probList"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const previewBox = document.getElementById('previewBox');
        const previewImg = document.getElementById('previewImg');
        const diagnoseBtn = document.getElementById('diagnoseBtn');
        const errorBanner = document.getElementById('errorBanner');
        const resContainer = document.getElementById('resContainer');
        const statusBox = document.getElementById('statusBox');
        const tabContent1 = document.getElementById('tabContent1');
        const tabContent2 = document.getElementById('tabContent2');
        const probList = document.getElementById('probList');
        const tabBtn1 = document.getElementById('tabBtn1');
        const tabBtn2 = document.getElementById('tabBtn2');
        const expanderContent = document.getElementById('expanderContent');
        const expanderArrow = document.getElementById('expanderArrow');

        let selectedFile = null;

        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            selectedFile = file;
            errorBanner.style.display = 'none';
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                previewBox.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        diagnoseBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                errorBanner.textContent = "Please upload an image before clicking 'Diagnose Disease'.";
                errorBanner.style.display = 'block';
                return;
            }

            errorBanner.style.display = 'none';
            diagnoseBtn.disabled = true;
            diagnoseBtn.innerHTML = '<span class="spinner"></span>Analyzing image, please wait...';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/api/diagnose', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                statusBox.className = 'status-box status-' + data.status_type;
                statusBox.innerHTML = 'Detected Condition: ' + data.name;

                tabContent1.innerHTML = data.immediate;
                tabContent2.innerHTML = data.chemical;

                let probsHtml = '<div style="margin-bottom: 0.5rem; color: #8b949e;">Predicted Class Confidence Distribution:</div>';
                for (const [cls, pct] of Object.entries(data.probabilities)) {
                    probsHtml += `<div class="prob-row"><span>${cls}:</span><span><strong>${pct}%</strong></span></div>`;
                }
                probList.innerHTML = probsHtml;

                resContainer.style.display = 'block';
            } catch (err) {
                errorBanner.textContent = "Error running analysis: " + err.message;
                errorBanner.style.display = 'block';
            } finally {
                diagnoseBtn.disabled = false;
                diagnoseBtn.innerHTML = 'Diagnose Disease';
            }
        });

        function switchTab(num) {
            if (num === 1) {
                tabBtn1.classList.add('active');
                tabBtn2.classList.remove('active');
                tabContent1.classList.add('active');
                tabContent2.classList.remove('active');
            } else {
                tabBtn1.classList.remove('active');
                tabBtn2.classList.add('active');
                tabContent1.classList.remove('active');
                tabContent2.classList.add('active');
            }
        }

        function toggleExpander() {
            if (expanderContent.style.display === 'block') {
                expanderContent.style.display = 'none';
                expanderArrow.innerHTML = '&#9662;';
            } else {
                expanderContent.style.display = 'block';
                expanderArrow.innerHTML = '&#9652;';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_tensor = transform(img)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.inference_mode():
        output = model(img_tensor)

    pred = torch.argmax(output, axis=1).item()
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    confidence = round(probabilities[pred].item() * 100, 2)
    label = label_encoder.inverse_transform([pred])[0]
    res = disease_details[label]

    prob_dict = {}
    for idx, prob in enumerate(probabilities):
        class_name = label_encoder.inverse_transform([idx])[0]
        readable = disease_details.get(class_name, {}).get('name', class_name)
        prob_dict[readable] = f"{prob.item() * 100:.2f}"

    return JSONResponse(content={
        "name": res["name"],
        "status_type": res["status_type"],
        "confidence": confidence,
        "immediate": res["immediate"],
        "chemical": res["chemical"],
        "probabilities": prob_dict
    })

def get_available_port(start_port=8501):
    import socket
    for p in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return start_port

if __name__ == "__main__":
    port = get_available_port(8501)
    if port != 8501:
        print(f"\nNotice: Port 8501 was in use. Launching on http://localhost:{port} instead.\n")
    else:
        print(f"\nStarting server on http://localhost:{port} ...\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
