document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    
    if (!file) {
        alert('Please select an image first');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed');
    }
});

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    const content = document.getElementById('resultsContent');
    
    const html = '<div class="result-item">' +
        '<h3>Vehicle Information</h3>' +
        '<p><strong>Make:</strong> ' + results.vehicle.make + '</p>' +
        '<p><strong>Model:</strong> ' + results.vehicle.model + '</p>' +
        '<p><strong>Type:</strong> ' + results.vehicle.type + '</p>' +
        '<p><strong>Year:</strong> ' + results.vehicle.year + '</p>' +
        '<p><strong>Confidence:</strong> ' + (results.vehicle.confidence * 100).toFixed(1) + '%</p>' +
        '</div>' +
        '<div class="result-item">' +
        '<h3>Color</h3>' +
        '<p><strong>Detected:</strong> ' + results.color.prediction + '</p>' +
        '<p><strong>Confidence:</strong> ' + (results.color.confidence * 100).toFixed(1) + '%</p>' +
        '</div>' +
        '<div class="result-item">' +
        '<h3>OOD Detection</h3>' +
        '<p>' + (results.is_ood ? '⚠️ Out-of-Distribution' : '✓ In-Distribution') + '</p>' +
        '</div>';
    
    content.innerHTML = html;
    resultsDiv.classList.remove('hidden');
}