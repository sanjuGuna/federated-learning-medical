document.addEventListener('DOMContentLoaded', () => {
    
    // Dataset Switching Logic
    const datasetSelector = document.getElementById('dataset-selector');
    const forms = document.querySelectorAll('.dataset-form');
    const metricsTitle = document.getElementById('metrics-dataset-title');

    datasetSelector.addEventListener('change', (e) => {
        const selected = e.target.value;
        
        // Hide all forms, show selected
        forms.forEach(f => f.classList.add('hidden'));
        document.getElementById(`form-${selected}`).classList.remove('hidden');
        
        // Update metrics title
        metricsTitle.textContent = selected;
        
        // Fetch metrics for selected dataset
        fetchMetrics(selected);
        
        // Reset prediction UI
        resetPredictionUI();
    });

    // Update age displays
    const ageInputs = [
        {input: 'age-diab', val: 'age-val-diab'},
        {input: 'age-hcv', val: 'age-val-hcv'},
        {input: 'age-derm', val: 'age-val-derm'}
    ];
    
    ageInputs.forEach(item => {
        const el = document.getElementById(item.input);
        if (el) {
            el.addEventListener('input', (e) => {
                document.getElementById(item.val).textContent = e.target.value;
            });
        }
    });

    // Fetch and display metrics on load for default dataset (diabetes)
    fetchMetrics('diabetes');

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
            const selectedDataset = datasetSelector.value;
            const formData = new FormData(form);
            let payload = {};
            
            // Build payload exactly as backend expects based on dataset
            if (selectedDataset === 'diabetes') {
                payload = {
                    Age: parseFloat(document.getElementById('age-diab').value),
                    Gender: formData.get('Gender'),
                    Polyuria: formData.getAll('Polyuria').length ? "Yes" : "No",
                    Polydipsia: formData.getAll('Polydipsia').length ? "Yes" : "No",
                    sudden_weight_loss: formData.getAll('sudden weight loss').length ? "Yes" : "No",
                    weakness: formData.getAll('weakness').length ? "Yes" : "No",
                    Polyphagia: formData.getAll('Polyphagia').length ? "Yes" : "No",
                    Genital_thrush: formData.getAll('Genital thrush').length ? "Yes" : "No",
                    visual_blurring: formData.getAll('visual blurring').length ? "Yes" : "No",
                    Itching: formData.getAll('Itching').length ? "Yes" : "No",
                    Irritability: formData.getAll('Irritability').length ? "Yes" : "No",
                    delayed_healing: formData.getAll('delayed healing').length ? "Yes" : "No",
                    partial_paresis: formData.getAll('partial paresis').length ? "Yes" : "No",
                    muscle_stiffness: formData.getAll('muscle stiffness').length ? "Yes" : "No",
                    Alopecia: formData.getAll('Alopecia').length ? "Yes" : "No",
                    Obesity: formData.getAll('Obesity').length ? "Yes" : "No"
                };
            } else if (selectedDataset === 'hcv') {
                payload = {
                    Age: parseFloat(document.getElementById('age-hcv').value),
                    Sex: formData.get('Sex'),
                    ALB: parseFloat(formData.get('ALB')),
                    ALP: parseFloat(formData.get('ALP')),
                    ALT: parseFloat(formData.get('ALT')),
                    AST: parseFloat(formData.get('AST')),
                    BIL: parseFloat(formData.get('BIL')),
                    CHE: parseFloat(formData.get('CHE')),
                    CHOL: parseFloat(formData.get('CHOL')),
                    CREA: parseFloat(formData.get('CREA')),
                    GGT: parseFloat(formData.get('GGT')),
                    PROT: parseFloat(formData.get('PROT'))
                };
            } else if (selectedDataset === 'dermatology') {
                payload = {
                    Age: parseFloat(document.getElementById('age-derm').value),
                    F11: parseFloat(formData.get('F11'))
                };
                for (let i = 1; i <= 33; i++) {
                    if (i !== 11) {
                        payload[`F${i}`] = parseFloat(formData.get(`F${i}`));
                    }
                }
            }

            const response = await fetch(`/predict/${selectedDataset}`, {
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

function resetPredictionUI() {
    document.getElementById('pred-class').textContent = '--';
    document.getElementById('pred-conf').textContent = '--% confidence';
    const predBar = document.getElementById('pred-bar');
    predBar.style.width = '0%';
    predBar.className = 'progress-bar';
    document.getElementById('neighbors-list').innerHTML = '<p class="placeholder-text">Run diagnosis to see similar patients</p>';
}

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

    // Highlight generic positive/negative or use generic accent
    if (data.prediction === 'Positive') {
        predClass.classList.add('positive-pred');
        predBar.classList.add('positive-bar');
    } else if (data.prediction === 'Negative') {
        predClass.classList.add('negative-pred');
        predBar.classList.add('negative-bar');
    } else {
        // Multi-class neutral highlighting
        predClass.style.color = 'var(--text-primary)';
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

async function fetchMetrics(dataset) {
    const cardsContainer = document.getElementById('metrics-cards');
    cardsContainer.innerHTML = '<div class="loading-metrics">Loading metrics...</div>';
    
    try {
        const response = await fetch(`/metrics?dataset=${dataset}`);
        if (!response.ok) return;
        const data = await response.json();
        
        // Populate Ablation Table
        const tbody = document.getElementById('ablation-body');
        tbody.innerHTML = '';
        document.getElementById('best-model-banner').classList.add('hidden');
        
        if (data.ablation && data.ablation.length > 0) {
            
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

            // Populate Metric Cards (using HYBRID metrics or Best model metrics)
            const displayMetrics = hybridMetrics || bestModel;
            if (displayMetrics) {
                cardsContainer.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-val">${displayMetrics.Accuracy}%</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${displayMetrics.Precision}%</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${displayMetrics.Recall}%</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${displayMetrics['F1-Score']}%</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-val">${displayMetrics['ROC-AUC']}</div>
                        <div class="metric-label">ROC-AUC</div>
                    </div>
                `;
            }
        } else {
            cardsContainer.innerHTML = '<div class="placeholder-text">No metrics found.</div>';
        }

    } catch (e) {
        console.error("Failed to fetch metrics", e);
        cardsContainer.innerHTML = '<div class="placeholder-text">Failed to load metrics. Ensure backend models are trained.</div>';
    }
}
