/**
 * TEACHABLE CLONE - Logic using TensorFlow.js & KNN Classifier
 */

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

// State
let counts = { 0: 0, 1: 0 };
let isTraining = { 0: false, 1: false };
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
        if (classifier.getNumClasses() > 0) {
            const result = await tf.tidy(() => {
                const img = tf.browser.fromPixels(webcamElement);
                const activation = net.infer(img, 'conv_preds');
                return classifier.predictClass(activation);
            });
            updateUI(result);
        }

        if (isTraining[0] || isTraining[1]) {
            const i = isTraining[0] ? 0 : 1;
            tf.tidy(() => {
                const img = tf.browser.fromPixels(webcamElement);
                const activation = net.infer(img, 'conv_preds');
                classifier.addExample(activation, i);
            });
            counts[isTraining[0] ? 0 : 1]++;
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

    // Bar A
    document.getElementById('prob-a-bar').style.width = `${probA * 100}%`;
    document.getElementById('prob-a-text').innerText = `${Math.round(probA * 100)}%`;
    
    // Bar B
    document.getElementById('prob-b-bar').style.width = `${probB * 100}%`;
    document.getElementById('prob-b-text').innerText = `${Math.round(probB * 100)}%`;

    // Highlight Winner
    if (probA > 0.6) {
        document.getElementById('class-a-card').style.borderColor = '#3b82f6';
        document.getElementById('class-b-card').style.borderColor = '#f1f5f9';
    } else if (probB > 0.6) {
        document.getElementById('class-b-card').style.borderColor = '#6366f1';
        document.getElementById('class-a-card').style.borderColor = '#f1f5f9';
    }
}

function updateCountsUI() {
    document.getElementById('class-a-count').innerText = `${counts[0]} amostras`;
    document.getElementById('class-b-count').innerText = `${counts[1]} amostras`;
}

// Event Listeners for Training
const setupButtons = () => {
    // Class A
    const btnA = document.getElementById('class-a-btn');
    const cardA = document.getElementById('class-a-card');
    
    btnA.addEventListener('mousedown', () => { isTraining[0] = true; cardA.classList.add('recording'); });
    btnA.addEventListener('mouseup', () => { isTraining[0] = false; cardA.classList.remove('recording'); });
    btnA.addEventListener('mouseleave', () => { isTraining[0] = false; cardA.classList.remove('recording'); });
    
    // Touch support
    btnA.addEventListener('touchstart', (e) => { e.preventDefault(); isTraining[0] = true; cardA.classList.add('recording'); });
    btnA.addEventListener('touchend', () => { isTraining[0] = false; cardA.classList.remove('recording'); });

    // Class B
    const btnB = document.getElementById('class-b-btn');
    const cardB = document.getElementById('class-b-card');
    
    btnB.addEventListener('mousedown', () => { isTraining[1] = true; cardB.classList.add('recording'); });
    btnB.addEventListener('mouseup', () => { isTraining[1] = false; cardB.classList.remove('recording'); });
    btnB.addEventListener('mouseleave', () => { isTraining[1] = false; cardB.classList.remove('recording'); });
    
    btnB.addEventListener('touchstart', (e) => { e.preventDefault(); isTraining[1] = true; cardB.classList.add('recording'); });
    btnB.addEventListener('touchend', () => { isTraining[1] = false; cardB.classList.remove('recording'); });

    // Reset
    document.getElementById('reset-btn').onclick = () => {
        classifier.clearAllClasses();
        counts = { 0: 0, 1: 0 };
        updateCountsUI();
        document.getElementById('prediction-container').classList.add('opacity-50', 'grayscale');
    };

    // Dynamic Label Sync
    const inputA = document.querySelector('#class-a-card input');
    const inputB = document.querySelector('#class-b-card input');
    
    inputA.addEventListener('input', (e) => document.getElementById('label-a-text').innerText = e.target.value);
    inputB.addEventListener('input', (e) => document.getElementById('label-b-text').innerText = e.target.value);
};

setupButtons();
app().catch(err => {
    console.error(err);
    document.getElementById('status-badge').innerHTML = "❌ Erro ao carregar";
});
