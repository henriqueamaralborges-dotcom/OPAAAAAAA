/**
 * TEACHABLE CLONE - Logic using TensorFlow.js & KNN Classifier
 */

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

// State
let counts = { 0: 0, 1: 0, 2: 0 };
let isTraining = { 0: false, 1: false, 2: false };
let isPredicting = false; // Toggle state
let modelLoaded = false;

async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user' }, 
            audio: false 
        });
        webcamElement.srcObject = stream;
        return new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => {
                document.getElementById('loader').style.opacity = '0';
                setTimeout(() => document.getElementById('loader').style.display = 'none', 500);
                resolve();
            };
        });
    } catch (err) {
        console.error("Webcam Error:", err);
        document.getElementById('status-badge').innerHTML = "⚠️ Erro de Câmera";
        document.getElementById('status-badge').classList.replace('bg-amber-100', 'bg-red-100');
        alert("Erro ao acessar a câmera. Certifique-se de estar em um ambiente seguro (HTTPS) e ter dado permissão.");
        throw err;
    }
}

async function app() {
    console.log('Loading mobilenet..');
    
    // Load the model
    net = await mobilenet.load();
    console.log('Successfully loaded model');
    modelLoaded = true;
    
    // UI Update
    document.getElementById('status-badge').innerHTML = `
        <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
        Modelos Prontos
    `;
    document.getElementById('status-badge').classList.replace('bg-amber-100', 'bg-emerald-100');
    document.getElementById('status-badge').classList.replace('text-amber-700', 'text-emerald-700');
    
    // Enable Buttons
    document.getElementById('class-a-btn').disabled = false;
    document.getElementById('class-b-btn').disabled = false;

    await setupWebcam();

    // Loop de Previsão
    while (true) {
        if (isPredicting && classifier.getNumClasses() > 0) {
            const activation = net.infer(webcamElement, 'conv_preds');
            const result = await classifier.predictClass(activation);
            updateUI(result);
        }

        if (isTraining[0] || isTraining[1] || isTraining[2]) {
            let i = isTraining[0] ? 0 : (isTraining[1] ? 1 : 2);
            const activation = net.infer(webcamElement, 'conv_preds');
            classifier.addExample(activation, i);
            counts[i]++;
            updateCountsUI();
        }

        await tf.nextFrame();
    }
}

// UI Handlers
function updateUI(result) {
    const container = document.getElementById('prediction-container');
    container.classList.remove('opacity-50', 'grayscale');

    const probA = result.confidences[0] || 0;
    const probB = result.confidences[1] || 0;
    const probC = result.confidences[2] || 0;

    // Bar A
    document.getElementById('prob-a-bar').style.width = `${probA * 100}%`;
    document.getElementById('prob-a-text').innerText = `${Math.round(probA * 100)}%`;
    
    // Bar B
    document.getElementById('prob-b-bar').style.width = `${probB * 100}%`;
    document.getElementById('prob-b-text').innerText = `${Math.round(probB * 100)}%`;

    // Bar C
    document.getElementById('prob-c-bar').style.width = `${probC * 100}%`;
    document.getElementById('prob-c-text').innerText = `${Math.round(probC * 100)}%`;

    // Final Verdict Logic
    const verdictText = document.getElementById('verdict-text');
    const verdictBox = document.getElementById('verdict-box');
    const labelA = document.querySelector('#class-a-card input').value;
    const labelB = document.querySelector('#class-b-card input').value;
    const labelC = document.querySelector('#class-c-card input').value;

    // Use a higher threshold for non-neutral classes
    if (probC > 0.7 && probC > probA && probC > probB) {
        verdictText.innerText = labelC;
        verdictText.style.color = "#94a3b8";
        verdictBox.style.boxShadow = "none";
        verdictText.innerText = labelA;
        verdictText.style.color = "#3b82f6";
        verdictBox.style.boxShadow = "0 0 30px rgba(59, 130, 246, 0.4)";
        verdictBox.style.borderColor = "rgba(59, 130, 246, 0.5)";
    } else if (probB > 0.45 && probB > probA && probB > probC) {
        verdictText.innerText = labelB;
        verdictText.style.color = "#6366f1";
        verdictBox.style.boxShadow = "0 0 30px rgba(99, 102, 241, 0.4)";
        verdictBox.style.borderColor = "rgba(99, 102, 241, 0.5)";
    } else {
        verdictText.innerText = "Analisando...";
        verdictText.style.color = "#94a3b8";
        verdictBox.style.boxShadow = "none";
        verdictBox.style.borderColor = "transparent";
    }
}

function updateCountsUI() {
    document.getElementById('class-a-count').innerText = `${counts[0]} amostras`;
    document.getElementById('class-b-count').innerText = `${counts[1]} amostras`;
    document.getElementById('class-c-count').innerText = `${counts[2]} amostras`;
}

// Event Listeners for Training
const setupButtons = () => {
    // Buttons for 3 Classes
    [0, 1, 2].forEach(id => {
        const char = String.fromCharCode(97 + id); // a, b, c
        const btn = document.getElementById(`class-${char}-btn`);
        const card = document.getElementById(`class-${char}-card`);
        const upload = document.getElementById(`class-${char}-upload`);
        
        if (btn) {
            btn.disabled = false;
            btn.addEventListener('mousedown', () => { isTraining[id] = true; card.classList.add('recording'); });
            btn.addEventListener('mouseup', () => { isTraining[id] = false; card.classList.remove('recording'); });
            btn.addEventListener('mouseleave', () => { isTraining[id] = false; card.classList.remove('recording'); });
            btn.addEventListener('touchstart', (e) => { e.preventDefault(); isTraining[id] = true; card.classList.add('recording'); });
            btn.addEventListener('touchend', () => { isTraining[id] = false; card.classList.remove('recording'); });
        }

        if (upload) {
            upload.addEventListener('change', (e) => handleUpload(e, id));
        }

        const input = document.querySelector(`#class-${char}-card input`);
        if (input) {
            input.addEventListener('input', (e) => {
                const label = document.getElementById(`label-${char}-text`);
                if (label) label.innerText = e.target.value;
            });
        }
    });

    // Shared Upload Logic
    async function handleUpload(event, classId) {
        const files = event.target.files;
        if (!files.length) return;
        
        const badge = document.getElementById('status-badge');
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-blue-500 animate-spin"></span> Processando...`;
        
        for (const file of files) {
            const img = await loadImage(file);
            const activation = net.infer(img, 'conv_preds');
            classifier.addExample(activation, classId);
            counts[classId]++;
            updateCountsUI();
            await tf.nextFrame(); 
        }
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Sucesso (+${files.length})`;
        event.target.value = '';
    }

    function loadImage(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    // Test Upload
    document.getElementById('test-upload').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (classifier.getNumClasses() === 0) {
            alert("⚠️ Treine a IA primeiro!");
            return;
        }

        const img = await loadImage(file);
        const activation = net.infer(img, 'conv_preds');
        const res = await classifier.predictClass(activation);
        updateUI(res);
        e.target.value = '';
    });

    // Export
    document.getElementById('export-btn').onclick = () => {
        const dataset = classifier.getClassifierDataset();
        const datasetObj = {};
        Object.keys(dataset).forEach((key) => {
            const data = dataset[key].dataSync();
            datasetObj[key] = Array.from(data);
        });
        const blob = new Blob([JSON.stringify(datasetObj)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'modelo_ia.json';
        a.click();
    };

    // Import
    document.getElementById('import-upload').onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            try {
                const datasetJson = JSON.parse(event.target.result);
                const dataset = {};
                Object.keys(datasetJson).forEach((key) => {
                    dataset[key] = tf.tensor2d(datasetJson[key], [datasetJson[key].length / 1024, 1024]);
                });
                classifier.setClassifierDataset(dataset);
                
                // Recalculate counts
                counts = { 0: 0, 1: 0, 2: 0 };
                Object.keys(dataset).forEach(key => {
                    counts[key] = dataset[key].shape[0];
                });
                updateCountsUI();
                
                document.getElementById('status-badge').innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Modelo Importado!`;
                alert("Modelo carregado com sucesso!");
            } catch (err) {
                console.error(err);
                alert("Erro ao importar modelo. Verifique se o arquivo JSON é válido.");
            }
        };
        reader.readAsText(file);
    };

    // Toggle Prediction Button
    const toggleBtn = document.getElementById('toggle-predict-btn');
    toggleBtn.onclick = () => {
        if (classifier.getNumClasses() === 0) {
            alert("⚠️ Treine a IA primeiro com algumas fotos ou pela webcam!");
            return;
        }
        
        isPredicting = !isPredicting;
        const btnText = document.getElementById('btn-text');
        const btnIcon = document.getElementById('btn-icon');
        const badge = document.getElementById('status-badge');
        
        if (isPredicting) {
            toggleBtn.classList.replace('bg-blue-600', 'bg-red-600');
            toggleBtn.classList.replace('hover:bg-blue-700', 'hover:bg-red-700');
            btnText.innerText = "Parar Análise";
            btnIcon.innerText = "⏹";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-blue-500 animate-ping"></span> IA Analisando Transmissão...`;
        } else {
            toggleBtn.classList.replace('bg-red-600', 'bg-blue-600');
            toggleBtn.classList.replace('hover:bg-red-700', 'hover:bg-blue-700');
            btnText.innerText = "Iniciar Análise";
            btnIcon.innerText = "▶";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Análise Pausada`;
            document.getElementById('prediction-container').classList.add('opacity-50', 'grayscale');
            document.getElementById('verdict-text').innerText = "Pausado";
        }
    };

    // Reset Logic
    document.getElementById('reset-btn').onclick = () => {
        if (confirm("Tem certeza que deseja apagar todo o treinamento?")) {
            classifier.clearAllClasses();
            counts = { 0: 0, 1: 0, 2: 0 };
            updateCountsUI();
            isPredicting = false;
            // UI Reset
            toggleBtn.classList.replace('bg-red-600', 'bg-blue-600');
            document.getElementById('btn-text').innerText = "Iniciar Análise";
            document.getElementById('btn-icon').innerText = "▶";
            document.getElementById('prediction-container').classList.add('opacity-50', 'grayscale');
            document.getElementById('verdict-text').innerText = "Reiniciado";
            document.getElementById('status-badge').innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500"></span> Dados Limpos`;
        }
    };
};

setupButtons();
app().catch(err => {
    console.error(err);
    document.getElementById('status-badge').innerHTML = "❌ Erro";
});

