async function detectObjects() {
    const imageInput = document.getElementById('imageInput');
    const feature1Input = document.getElementById('feature1');
    const feature2Input = document.getElementById('feature2');
    const loading = document.getElementById('loading');
    const canvas = document.getElementById('outputCanvas');
    const classCountsContainer = document.getElementById('classCountsContainer');
    const yearsValue = document.getElementById('yearsValue');

    if (!imageInput.files[0]) {
        alert('Please select an image first');
        return;
    }

    const feature1 = feature1Input.value;
    const feature2 = feature2Input.value;

    if (!feature1 || !feature2) {
        alert('Please enter values for Feature 1 and Feature 2');
        return;
    }

    loading.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        formData.append('feature1', feature1);
        formData.append('feature2', feature2);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Draw image and bounding boxes
            await drawDetections(canvas, result.image, result.detections);
            
            // Display class counts
            displayClassCounts(result.class_counts);
            
            // Display years replacement prediction
            displayYearsPrediction(result.years_replacement);
        } else {
            alert('Error processing image: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        loading.classList.add('hidden');
    }
}

async function drawDetections(canvas, base64Image, detections) {
    const ctx = canvas.getContext('2d');
    
    // Load and draw the image
    const img = new Image();
    img.src = 'data:image/jpeg;base64,' + base64Image;
    
    await new Promise((resolve) => {
        img.onload = () => {
            // Set canvas size to match image
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Draw image
            ctx.drawImage(img, 0, 0);
            
            // Draw detection boxes
            detections.forEach(detection => {
                const [x1, y1, x2, y2] = detection.box;
                const width = x2 - x1;
                const height = y2 - y1;
                
                // Draw rectangle
                ctx.strokeStyle = '#2ecc71';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, width, height);
                
                // Draw label background
                const label = `${detection.class} ${Math.round(detection.confidence * 100)}%`;
                ctx.font = '14px Arial';
                const textWidth = ctx.measureText(label).width;
                ctx.fillStyle = '#2ecc71';
                ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);
                
                // Draw label text
                ctx.fillStyle = 'white';
                ctx.fillText(label, x1 + 5, y1 - 5);
            });
            
            resolve();
        };
    });
}

function displayClassCounts(classCounts) {
    const container = document.getElementById('classCountsContainer');
    container.innerHTML = '';
    
    Object.entries(classCounts).forEach(([className, count]) => {
        const item = document.createElement('div');
        item.className = 'class-count-item';
        item.innerHTML = `
            <span>${className}:</span>
            <span>${count}</span>
        `;
        container.appendChild(item);
    });
}

function displayYearsPrediction(years) {
    const yearsValue = document.getElementById('yearsValue');
    yearsValue.textContent = years.toFixed(1) + ' years';
}

// Add file input change handler to show selected filename
document.getElementById('imageInput').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Choose Image';
    document.querySelector('.file-input-label').textContent = fileName;
});