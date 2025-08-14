async function getJSON(url) {
	const res = await fetch(url);
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

async function postForm(url, formData) {
	const res = await fetch(url, { method: 'POST', body: formData });
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

async function postJSON(url, params = {}) {
	const res = await fetch(url + (params ? '?' + new URLSearchParams(params) : ''), { method: 'POST' });
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

function setPre(id, obj) {
	document.getElementById(id).textContent = JSON.stringify(obj, null, 2);
}

async function loadInfo() {
	try {
		setPre('models', await getJSON('/api/models'));
		setPre('classes', await getJSON('/api/classes'));
		setPre('config', await getJSON('/api/config'));
	} catch (e) {
		console.error(e);
	}
}

function setupPredict() {
	const input = document.getElementById('predict-file');
	const preview = document.getElementById('preview');
	input.addEventListener('change', () => {
		const file = input.files?.[0];
		if (file) {
			preview.src = URL.createObjectURL(file);
		}
	});
	document.getElementById('predict-form').addEventListener('submit', async (e) => {
		e.preventDefault();
		const file = input.files?.[0];
		if (!file) return;
		const fd = new FormData();
		fd.append('file', file);
		try { setPre('predict-result', await postForm('/api/predict', fd)); }
		catch (err) { setPre('predict-result', { error: err.message }); }
	});
}

function setupTrain() {
	document.getElementById('train-form').addEventListener('submit', async (e) => {
		e.preventDefault();
		const model = document.getElementById('train-model').value;
		setPre('train-result', { status: 'training started...' });
		try {
			const res = await postJSON('/api/train', { model });
			setPre('train-result', res);
			await loadRuns();
		} catch (err) {
			setPre('train-result', { error: err.message });
		}
	});
}

async function loadRuns() {
	const ul = document.getElementById('runs-list');
	ul.innerHTML = '';
	try {
		const data = await getJSON('/api/runs');
		for (const run of data.runs) {
			const li = document.createElement('li');
			li.innerHTML = `<span>${run.name} (${new Date(run.mtime*1000).toLocaleString()})</span><button data-run="${run.name}">View</button>`;
			li.querySelector('button').addEventListener('click', () => viewRun(run.name));
			ul.appendChild(li);
		}
	} catch (e) { console.error(e); }
}

let accChart, lossChart;
function renderCharts(history) {
	const epochs = history?.accuracy?.length ? Array.from({length: history.accuracy.length}, (_, i) => i+1) : [];
	const ctxA = document.getElementById('accChart').getContext('2d');
	const ctxL = document.getElementById('lossChart').getContext('2d');
	if (accChart) accChart.destroy();
	if (lossChart) lossChart.destroy();
	accChart = new Chart(ctxA, {
		type: 'line',
		data: { labels: epochs, datasets: [
			{ label: 'accuracy', data: history.accuracy || [], borderColor: '#0b5' },
			{ label: 'val_accuracy', data: history.val_accuracy || [], borderColor: '#07c' }
		]},
		options: { responsive: true, maintainAspectRatio: false }
	});
	lossChart = new Chart(ctxL, {
		type: 'line',
		data: { labels: epochs, datasets: [
			{ label: 'loss', data: history.loss || [], borderColor: '#f66' },
			{ label: 'val_loss', data: history.val_loss || [], borderColor: '#b40' }
		]},
		options: { responsive: true, maintainAspectRatio: false }
	});
}

async function viewRun(name) {
	try {
		const data = await getJSON(`/api/runs/${encodeURIComponent(name)}/history`);
		renderCharts(data.history || {});
	} catch (e) { console.error(e); }
}

(async function init() {
	await loadInfo();
	setupPredict();
	setupTrain();
	await loadRuns();
})();