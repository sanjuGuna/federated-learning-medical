document.addEventListener('DOMContentLoaded', () => {
    
    // Update age display
    const ageInput = document.getElementById('age');
    const ageVal = document.getElementById('age-val');
    ageInput.addEventListener('input', (e) => {
        ageVal.textContent = e.target.value;
    });

    // Fetch and display metrics on load
    fetchMetrics();

    // Handle form submission
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const runBtn = document.getElementById('run-btn');
        const btnText = runBtn.querySelector('.btn-text');
        const spinner = document.getElementById('loading-spinner');
        
        // Start loading state
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        runBtn.disabled = true;

        try {
            // Serialize form data
            const formData = new FormData(form);
            
            // Build payload exactly as backend expects
            const payload = {
                Age: parseFloat(formData.get('Age')),
                Gender: formData.get('Gender'),
                Polyuria: formData.get('Polyuria') ? "Yes" : "No",
                Polydipsia: formData.get('Polydipsia') ? "Yes" : "No",
                sudden_weight_loss: formData.get('sudden weight loss') ? "Yes" : "No",
                weakness: formData.get('weakness') ? "Yes" : "No",
                Polyphagia: formData.get('Polyphagia') ? "Yes" : "No",
                Genital_thrush: formData.get('Genital thrush') ? "Yes" : "No",
                visual_blurring: formData.get('visual blurring') ? "Yes" : "No",
                Itching: formData.get('Itching') ? "Yes" : "No",
                Irritability: formData.get('Irritability') ? "Yes" : "No",
                delayed_healing: formData.get('delayed healing') ? "Yes" : "No",
                partial_paresis: formData.get('partial paresis') ? "Yes" : "No",
                muscle_stiffness: formData.get('muscle stiffness') ? "Yes" : "No",
                Alopecia: formData.get('Alopecia') ? "Yes" : "No",
                Obesity: formData.get('Obesity') ? "Yes" : "No"
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error("Failed to get prediction");
            }

            const data = await response.json();
            updatePredictionUI(data);

        } catch (error) {
            console.error(error);
            alert("Error running diagnosis. Ensure backend is running.");
        } finally {
            // Restore button state
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            runBtn.disabled = false;
        }
    });

});

function updatePredictionUI(data) {
    // 1. Update Prediction Bar
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const predBar = document.getElementById('pred-bar');

    predClass.textContent = data.prediction;
    predConf.textContent = `${data.confidence.toFixed(1)}% confidence`;
    
    // reset classes
    predClass.className = '';
    predBar.className = 'progress-bar';

    if (data.prediction === 'Positive') {
        predClass.classList.add('positive-pred');
        predBar.classList.add('positive-bar');
    } else {
        predClass.classList.add('negative-pred');
        predBar.classList.add('negative-bar');
    }

    // Trigger reflow for animation
    void predBar.offsetWidth;
    predBar.style.width = `${data.confidence}%`;

    // 2. Update Neighbors
    const neighborsList = document.getElementById('neighbors-list');
    neighborsList.innerHTML = '';
    
    if (data.neighbor_ids && data.neighbor_importance) {
        // Sort neighbors by importance
        const sortedNeighbors = data.neighbor_ids.map(id => {
            return { id, weight: data.neighbor_importance[`Neighbor_${id}`] || 0 };
        }).sort((a, b) => b.weight - a.weight);

        sortedNeighbors.forEach((n, i) => {
            const tag = document.createElement('div');
            tag.className = 'neighbor-tag';
            tag.style.animationDelay = `${i * 0.1}s`;
            tag.innerHTML = `Patient #${n.id} <span class="neighbor-attn">attn ${(n.weight).toFixed(2)}</span>`;
            neighborsList.appendChild(tag);
        });
    }


}

async function fetchMetrics() {
    try {
        const response = await fetch('/metrics');
        if (!response.ok) return;
        const data = await response.json();
        
        // Populate Ablation Table
        if (data.ablation && data.ablation.length > 0) {
            const tbody = document.getElementById('ablation-body');
            tbody.innerHTML = '';
            
            let hybridMetrics = null;
            let bestModel = null;

            data.ablation.forEach(row => {
                if (row.Model === 'HYBRID') hybridMetrics = row;
                
                // Track best model by Accuracy
                if (!bestModel || row.Accuracy > bestModel.Accuracy) {
                    bestModel = row;
                }
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${row.Model}</strong></td>
                    <td>${row.Accuracy}%</td>
                    <td>${row["F1-Score"]}%</td>
                    <td>${row["ROC-AUC"]}</td>
                `;
                tbody.appendChild(tr);
            });

            // Populate Best Model Banner
            if (bestModel) {
                document.getElementById('best-model-name').textContent = bestModel.Model;
                document.getElementById('best-model-acc').textContent = bestModel.Accuracy;
                document.getElementById('best-model-f1').textContent = bestModel["F1-Score"];
                document.getElementById('best-model-banner').classList.remove('hidden');
            }

            // Populate Metric Cards (using HYBRID metrics)
            if (hybridMetrics) {
                const cardsContainer = document.getElementById('metrics-cards');
                cardsContainer.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-val">${hybridMetrics.Accuracy}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${hybridMetrics.Precision}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${hybridMetrics.Recall}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${hybridMetrics['F1-Score']}%</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${hybridMetrics['ROC-AUC']}</div>
                        <div class="metric-label">ROC-AUC</div>
                    </div>
                `;
            }
        }

        // Populate Confusion Matrix
        if (data.confusion_matrix) {
            // cm format: [[TN, FP], [FN, TP]]
            const cm = data.confusion_matrix;
            document.getElementById('cm-tn').textContent = cm[0][0];
            document.getElementById('cm-fp').textContent = cm[0][1];
            document.getElementById('cm-fn').textContent = cm[1][0];
            document.getElementById('cm-tp').textContent = cm[1][1];
        }

    } catch (e) {
        console.error("Failed to fetch metrics", e);
        document.getElementById('metrics-cards').innerHTML = '<div class="placeholder-text">Failed to load metrics. Ensure backend models are trained.</div>';
    }
}
