const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const uploadBar = document.getElementById('upload-bar');
const uploadPercent = document.getElementById('upload-percent');
const removeBtn = document.getElementById('remove-btn');

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, e => {
        e.preventDefault();
        dropArea.classList.add('border-blue-500', 'bg-blue-50');
    });
});
['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, e => {
        e.preventDefault();
        dropArea.classList.remove('border-blue-500', 'bg-blue-50');
    });
});

dropArea.addEventListener('drop', e => {
    handleFiles(e.dataTransfer.files);
});
fileElem.addEventListener('change', () => {
    handleFiles(fileElem.files);
});
removeBtn.addEventListener('click', () => {
    resetPreview();
});

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (!file.name.endsWith('.csv')) {
            alert('Please upload a .csv file only.');
            return;
        }

        // Show preview
        filePreview.classList.remove('hidden');
        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
        uploadBar.style.width = '0%';
        uploadPercent.textContent = '0%';

        simulateUpload();
    }
}

function simulateUpload() {
    let progress = 0;
    const interval = setInterval(() => {
        if (progress >= 100) {
            clearInterval(interval);
        } else {
            progress += Math.floor(Math.random() * 10) + 5;
            progress = Math.min(progress, 100);
            uploadBar.style.width = `${progress}%`;
            uploadPercent.textContent = `${progress}%`;
        }
    }, 300);
}

function resetPreview() {
    filePreview.classList.add('hidden');
    fileElem.value = '';
}
