const form = document.getElementById('upload-form');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const loadingProgress = document.getElementById('loading-progress');
const fileInput = document.getElementById('image-input');
const uploadBubble = document.getElementById('upload-bubble');
const originalPreview = document.getElementById('original-preview');
const colorizedPreview = document.getElementById('colorized-preview');
let selectedFile = null; // will hold the uploaded file until server returns

// Custom cursor
const cursor = document.createElement('div');
cursor.style.width = '20px';
cursor.style.height = '20px';
cursor.style.borderRadius = '50%';
cursor.style.background = '#555';
cursor.style.position = 'fixed';
cursor.style.pointerEvents = 'none';
cursor.style.zIndex = 9999;
cursor.style.transition = 'transform 0.05s ease-out';
cursor.style.transform = 'translate(-50%, -50%)';
document.body.appendChild(cursor);

document.addEventListener('mousemove', e => {
  cursor.style.left = e.clientX + 'px';
  cursor.style.top = e.clientY + 'px';
});

// Bubble hover effect on upload
uploadBubble.addEventListener('mouseenter', () => {
  cursor.style.transform = 'translate(-50%, -50%) scale(1.5)';
});
uploadBubble.addEventListener('mouseleave', () => {
  cursor.style.transform = 'translate(-50%, -50%) scale(1)';
});

// Click to open file picker
// uploadBubble.addEventListener('click', () => fileInput.click());

// Drag & drop handling
uploadBubble.addEventListener('dragover', e => {
  e.preventDefault();
  uploadBubble.classList.add('dragover');
});

uploadBubble.addEventListener('dragleave', e => {
  e.preventDefault();
  uploadBubble.classList.remove('dragover');
});

uploadBubble.addEventListener('drop', e => {
  e.preventDefault();
  uploadBubble.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    selectedFile = files[0];
    console.log('File dropped:', selectedFile.name);
    // Auto-submit form on drop
    form.dispatchEvent(new Event('submit', {cancelable: true, bubbles: true}));
  }
});

// Handle file input change (when user selects from file picker)
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    selectedFile = fileInput.files[0];
    console.log('File selected:', selectedFile.name);
    form.dispatchEvent(new Event('submit', {cancelable: true, bubbles: true}));
  }
});

// Keep track of created object URLs to revoke them
let originalURL = null;
let colorizedURL = null;
let modelsList = document.getElementById('models-list');

// Fetch available models from backend and render checkboxes
async function loadModels() {
  try {
    const resp = await fetch('/models');
    if (!resp.ok) throw new Error('Failed to fetch models');
    const data = await resp.json();
    modelsList.innerHTML = '';
    data.models.forEach(m => {
      const id = 'model_' + m.name.replace(/[^a-z0-9]/gi, '_');
      const wrapper = document.createElement('div');
      wrapper.className = 'model-checkbox';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = id;
      cb.name = 'models';
      cb.value = m.name;
      // auto-check loaded models by default
      if (m.loaded) cb.checked = true;
      const label = document.createElement('label');
      label.htmlFor = id;
      label.textContent = m.display_name + (m.loaded ? '' : ' (failed to load)');
      wrapper.appendChild(cb);
      wrapper.appendChild(label);
      modelsList.appendChild(wrapper);
    });
  } catch (e) {
    modelsList.textContent = 'Could not load models';
    console.error(e);
  }
}

loadModels();

// Handle form submit
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const file = fileInput.files[0];
  if (!file) return;

  // Show original uploaded image
  const uploadedPreview = document.getElementById('uploaded-preview');
  uploadedPreview.src = URL.createObjectURL(file);

  // Show loading
  loadingDiv.style.display = 'block';
  loadingProgress.style.width = '0%';
  // do not show result area yet — wait until server returns
  // clear only the colorized preview while processing (don't remove DOM nodes)
  if (colorizedPreview) colorizedPreview.src = '';

  const formData = new FormData();
  formData.append('image', file);
  // Gather checked model names and append as repeated 'models' fields
  const checkboxes = document.querySelectorAll('#models-list input[type=checkbox]:checked');
  if (checkboxes.length === 0) {
    alert('Please select at least one model to run.');
    loadingDiv.style.display = 'none';
    return;
  }
  checkboxes.forEach(cb => formData.append('models', cb.value));

  try {    
    const response = await fetch('/colorize', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Upload failed');

    // Response is JSON with data-URIs
    const data = await response.json();
    // Hide loading
    loadingDiv.style.display = 'none';

    // Set original preview (model input grayscale returned by server)
    if (data.original) {
      originalPreview.src = data.original;
    }

    // Clear previous results
    const resultsList = document.getElementById('results-list');
    resultsList.innerHTML = '';

    (data.results || []).forEach(r => {
      const panel = document.createElement('div');
      panel.className = 'result-item';
      const title = document.createElement('div');
      title.className = 'result-title';
      title.textContent = r.name;
      const img = document.createElement('img');
      img.className = 'result-image';
      img.alt = r.name;
      img.src = r.image_b64;
      panel.appendChild(title);
      panel.appendChild(img);
      resultsList.appendChild(panel);
    });

    // Show results container
    resultDiv.style.display = 'flex';

  } catch (err) {
    console.error(err);
    loadingDiv.style.display = 'none';
    resultDiv.textContent = 'Error processing image.';
  }
});