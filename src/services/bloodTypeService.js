import * as tf from '@tensorflow/tfjs';

const OCR_HOST = process.env.REACT_APP_OCR_HOST || '127.0.0.1';
const OCR_PORT = process.env.REACT_APP_OCR_PORT || '4891';
const BLOOD_MODEL_BASE_URL = `http://${OCR_HOST}:${OCR_PORT}/blood-type-model`;
const BLOOD_MODEL_URL = `${BLOOD_MODEL_BASE_URL}/model.json`;
const CLASS_NAMES_URL = `${BLOOD_MODEL_BASE_URL}/class_names.json`;
const DEFAULT_LABELS = ['A', 'AB', 'B', 'O', 'null'];
const BLOOD_HINTS = ['blood', 'blood_type', 'bloodtype', 'type'];

let loadedModel = null;
let loadedLabels = DEFAULT_LABELS;
let loadError = null;
let backendReady = false;

async function ensureTfBackend() {
  if (backendReady) return;
  try {
    await tf.setBackend('cpu');
  } catch {
    // Keep default backend if CPU backend selection fails.
  }
  await tf.ready();
  backendReady = true;
}

function normalizeLabel(value) {
  return String(value || '').trim().toLowerCase();
}

function looksLikeBloodLabel(label) {
  const normalized = normalizeLabel(label);
  if (!normalized) return false;
  return BLOOD_HINTS.some((hint) => normalized.includes(hint));
}

function clampBox(box, width, height) {
  const x1 = Math.max(0, Math.min(width - 1, Math.round(box.x1 || 0)));
  const y1 = Math.max(0, Math.min(height - 1, Math.round(box.y1 || 0)));
  const x2 = Math.max(0, Math.min(width, Math.round(box.x2 || 0)));
  const y2 = Math.max(0, Math.min(height, Math.round(box.y2 || 0)));
  if (x2 <= x1 || y2 <= y1) return null;
  return { x1, y1, x2, y2 };
}

function boxArea(box) {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function getCandidateRegions(apiResult) {
  const regionBoxes = Array.isArray(apiResult?.regionBoxes) ? apiResult.regionBoxes : [];
  if (!regionBoxes.length) return [];

  const labeled = regionBoxes.filter((box) => looksLikeBloodLabel(box?.label));
  if (labeled.length) return labeled;

  const cls7 = regionBoxes.filter((box) => Number.isFinite(box?.cls) && Math.round(box.cls) === 7);
  if (cls7.length) return cls7;

  return [...regionBoxes].sort((a, b) => boxArea(b) - boxArea(a)).slice(0, 3);
}

function buildGlobalFallbackRegions(width, height) {
  const make = (rx1, ry1, rx2, ry2) => ({
    x1: Math.round(width * rx1),
    y1: Math.round(height * ry1),
    x2: Math.round(width * rx2),
    y2: Math.round(height * ry2)
  });

  return [
    // Typical NRC blood-type zone is right side, upper-middle rows.
    make(0.52, 0.18, 0.96, 0.46),
    make(0.55, 0.16, 0.98, 0.44),
    make(0.50, 0.22, 0.95, 0.50),
    make(0.58, 0.20, 0.98, 0.48)
  ];
}

function buildCandidateSubBoxes(box) {
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;
  if (w < 8 || h < 8) return [box];

  const make = (rx1, ry1, rx2, ry2) => ({
    x1: Math.round(box.x1 + w * rx1),
    y1: Math.round(box.y1 + h * ry1),
    x2: Math.round(box.x1 + w * rx2),
    y2: Math.round(box.y1 + h * ry2)
  });

  return [
    // Handwritten blood type is on the right side of the detected blood region.
    make(0.55, 0.00, 1.00, 1.00),
    make(0.60, 0.00, 1.00, 1.00),
    make(0.65, 0.00, 1.00, 1.00),
    make(0.55, 0.10, 1.00, 0.95),
    make(0.60, 0.10, 1.00, 0.95),
    make(0.65, 0.10, 1.00, 0.95),
    make(0.55, 0.20, 1.00, 1.00),
    make(0.60, 0.20, 1.00, 1.00),
    make(0.65, 0.20, 1.00, 1.00)
  ];
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load frame image for blood-type crop'));
    img.src = dataUrl;
  });
}

async function loadClassNames() {
  try {
    const response = await fetch(CLASS_NAMES_URL);
    if (!response.ok) return;
    const data = await response.json();
    if (Array.isArray(data) && data.length) {
      loadedLabels = data.map((v) => String(v));
    }
  } catch {
    // Keep defaults.
  }
}

async function loadModel() {
  if (loadedModel) return loadedModel;
  if (loadError) throw loadError;

  await ensureTfBackend();
  await loadClassNames();
  try {
    loadedModel = await tf.loadLayersModel(BLOOD_MODEL_URL);
  } catch (err) {
    try {
      loadedModel = await loadModelWithKeras3Patch();
    } catch (patchedErr) {
      loadError = new Error(`Unable to load blood_type_model_tfjs: ${patchedErr?.message || err?.message || 'unknown error'}`);
      throw loadError;
    }
  }
  return loadedModel;
}

function patchInputLayerShape(modelJson) {
  const cloned = JSON.parse(JSON.stringify(modelJson));
  const layers = cloned?.modelTopology?.model_config?.config?.layers;
  if (!Array.isArray(layers)) return cloned;

  for (const layer of layers) {
    if (layer?.class_name !== 'InputLayer') continue;
    const cfg = layer.config || {};
    if (cfg.batch_shape && !cfg.batch_input_shape) {
      cfg.batch_input_shape = cfg.batch_shape;
      delete cfg.batch_shape;
    }
    layer.config = cfg;
  }

  return cloned;
}

function tensorRefFromKerasTensor(value) {
  if (!value || typeof value !== 'object') return null;
  if (value.class_name !== '__keras_tensor__') return null;
  const history = value?.config?.keras_history;
  if (!Array.isArray(history) || history.length < 3) return null;
  const layerName = history[0];
  const nodeIndex = history[1];
  const tensorIndex = history[2];
  return [layerName, nodeIndex, tensorIndex, {}];
}

function collectTensorRefs(value, out) {
  const ref = tensorRefFromKerasTensor(value);
  if (ref) {
    out.push(ref);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((child) => collectTensorRefs(child, out));
  }
}

function patchInboundNodes(modelJson) {
  const cloned = JSON.parse(JSON.stringify(modelJson));
  const layers = cloned?.modelTopology?.model_config?.config?.layers;
  if (!Array.isArray(layers)) return cloned;

  for (const layer of layers) {
    const inbound = layer?.inbound_nodes;
    if (!Array.isArray(inbound) || !inbound.length) continue;
    if (Array.isArray(inbound[0])) continue;

    const patchedInbound = [];
    for (const nodeData of inbound) {
      if (Array.isArray(nodeData)) {
        patchedInbound.push(nodeData);
        continue;
      }
      const refs = [];
      const args = Array.isArray(nodeData?.args) ? nodeData.args : [];
      args.forEach((arg) => collectTensorRefs(arg, refs));
      patchedInbound.push(refs);
    }
    layer.inbound_nodes = patchedInbound;
  }

  return cloned;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url} (${response.status})`);
  }
  return response.json();
}

function concatUint8Arrays(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

async function loadModelWithKeras3Patch() {
  const rawJson = await fetchJson(BLOOD_MODEL_URL);
  const shapePatched = patchInputLayerShape(rawJson);
  const modelJson = patchInboundNodes(shapePatched);

  const weightsManifest = Array.isArray(modelJson.weightsManifest) ? modelJson.weightsManifest : [];
  const weightSpecs = [];
  const buffers = [];

  for (const group of weightsManifest) {
    if (Array.isArray(group.weights)) {
      weightSpecs.push(...group.weights);
    }
    const paths = Array.isArray(group.paths) ? group.paths : [];
    for (const relPath of paths) {
      const shardUrl = `${BLOOD_MODEL_BASE_URL}/${relPath}`;
      const shardRes = await fetch(shardUrl);
      if (!shardRes.ok) {
        throw new Error(`Failed to fetch weight shard ${relPath} (${shardRes.status})`);
      }
      const ab = await shardRes.arrayBuffer();
      buffers.push(new Uint8Array(ab));
    }
  }

  const weightData = concatUint8Arrays(buffers).buffer;
  const artifacts = {
    modelTopology: modelJson.modelTopology,
    format: modelJson.format || 'layers-model',
    generatedBy: modelJson.generatedBy || 'keras',
    convertedBy: modelJson.convertedBy || 'tfjs-converter',
    trainingConfig: modelJson.trainingConfig,
    userDefinedMetadata: modelJson.userDefinedMetadata,
    weightSpecs,
    weightData
  };

  const handler = {
    load: async () => artifacts
  };

  return tf.loadLayersModel(handler);
}

function inferInputSize(model) {
  const shape = model?.inputs?.[0]?.shape;
  if (Array.isArray(shape) && Number.isFinite(shape[1]) && Number.isFinite(shape[2])) {
    return { height: shape[1], width: shape[2] };
  }
  return { height: 224, width: 224 };
}

function preprocessVariant(sourceCtx, w, h, variantName) {
  const imageData = sourceCtx.getImageData(0, 0, w, h);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);

    let out = gray;
    if (variantName === 'high_contrast') {
      out = gray > 150 ? 255 : 0;
    } else if (variantName === 'invert_threshold') {
      out = gray > 145 ? 0 : 255;
    } else if (variantName === 'boost') {
      out = Math.max(0, Math.min(255, (gray - 128) * 1.8 + 128));
    }

    data[i] = out;
    data[i + 1] = out;
    data[i + 2] = out;
  }

  sourceCtx.putImageData(imageData, 0, 0);
}

function pickBestLabel(scores, labels) {
  if (!scores.length) return null;
  const ranked = scores
    .map((score, index) => ({ index, score: Number(score) }))
    .sort((a, b) => b.score - a.score);

  const best = ranked[0];
  if (!best) return null;

  const bestLabel = normalizeLabel(labels[best.index] || '');
  if (bestLabel !== 'null') {
    return { index: best.index, score: best.score };
  }

  const fallback = ranked.find((item) => normalizeLabel(labels[item.index] || '') !== 'null' && item.score >= 0.2);
  if (fallback) {
    return { index: fallback.index, score: fallback.score };
  }

  return { index: best.index, score: best.score };
}

function weightedSelectionScore(prediction, labels) {
  const label = normalizeLabel(labels[prediction.index] || '');
  if (label === 'null') return prediction.score * 0.45;
  return prediction.score;
}

function topScores(scores, labels, limit = 3) {
  const items = scores.map((score, index) => ({
    label: labels[index] || `class_${index}`,
    score: Number(score)
  }));
  items.sort((a, b) => b.score - a.score);
  return items.slice(0, limit);
}

function buildModelInputs(model, primaryInputTensor, secondaryInputTensor = null) {
  const inputCount = Array.isArray(model?.inputs) ? model.inputs.length : 1;
  if (inputCount <= 1) return primaryInputTensor;

  if (inputCount === 2) {
    return [primaryInputTensor, secondaryInputTensor || primaryInputTensor.clone()];
  }

  const inputs = [primaryInputTensor];
  for (let i = 1; i < inputCount; i += 1) {
    inputs.push((secondaryInputTensor || primaryInputTensor).clone());
  }
  return inputs;
}

async function predictScores(model, primaryCanvas, secondaryCanvas = null) {
  const outputTensor = tf.tidy(() => {
    const primaryInput = tf.browser.fromPixels(primaryCanvas).toFloat().div(255).expandDims(0);
    const secondaryInput = secondaryCanvas
      ? tf.browser.fromPixels(secondaryCanvas).toFloat().div(255).expandDims(0)
      : null;
    const modelInput = buildModelInputs(model, primaryInput, secondaryInput);
    const rawOutput = model.predict(modelInput);
    return Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
  });
  const scores = Array.from(await outputTensor.data());
  outputTensor.dispose();
  return scores;
}

export async function classifyBloodTypeFromFrame(dataUrl, apiResult) {
  if (!dataUrl || !apiResult) return null;

  const image = await loadImage(dataUrl);
  const imageW = image.naturalWidth;
  const imageH = image.naturalHeight;

  const detectedRegions = getCandidateRegions(apiResult)
    .map((r) => clampBox(r, imageW, imageH))
    .filter(Boolean);
  const fallbackRegions = buildGlobalFallbackRegions(imageW, imageH)
    .map((r) => clampBox(r, imageW, imageH))
    .filter(Boolean);
  const regions = detectedRegions.length ? detectedRegions : fallbackRegions;
  if (!regions.length) return null;

  const model = await loadModel();
  const { height: targetHeight, width: targetWidth } = inferInputSize(model);

  const variantNames = ['raw', 'high_contrast', 'invert_threshold', 'boost'];
  const focusCanvas = document.createElement('canvas');
  focusCanvas.width = targetWidth;
  focusCanvas.height = targetHeight;
  const focusCtx = focusCanvas.getContext('2d', { willReadFrequently: true });
  if (!focusCtx) return null;

  const regionCanvas = document.createElement('canvas');
  regionCanvas.width = targetWidth;
  regionCanvas.height = targetHeight;
  const regionCtx = regionCanvas.getContext('2d', { willReadFrequently: true });
  if (!regionCtx) return null;

  let best = null;
  for (const region of regions) {
    const regionBox = clampBox(region, imageW, imageH);
    if (!regionBox) continue;

    regionCtx.clearRect(0, 0, targetWidth, targetHeight);
    regionCtx.drawImage(
      image,
      regionBox.x1,
      regionBox.y1,
      regionBox.x2 - regionBox.x1,
      regionBox.y2 - regionBox.y1,
      0,
      0,
      targetWidth,
      targetHeight
    );

    const subBoxes = buildCandidateSubBoxes(regionBox)
      .map((b) => clampBox(b, imageW, imageH))
      .filter(Boolean);

    for (const candidate of subBoxes) {
      for (const variantName of variantNames) {
        focusCtx.clearRect(0, 0, targetWidth, targetHeight);
        focusCtx.drawImage(
          image,
          candidate.x1,
          candidate.y1,
          candidate.x2 - candidate.x1,
          candidate.y2 - candidate.y1,
          0,
          0,
          targetWidth,
          targetHeight
        );

        if (variantName !== 'raw') {
          preprocessVariant(focusCtx, targetWidth, targetHeight, variantName);
        }

        const scores = await predictScores(model, regionCanvas, focusCanvas);

        const picked = pickBestLabel(scores, loadedLabels);
        if (!picked) continue;

        const weightedScore = weightedSelectionScore(picked, loadedLabels);
        if (!best || weightedScore > best.weightedScore) {
          best = {
            picked,
            weightedScore,
            candidate,
            scores
          };
        }
      }
    }
  }

  if (!best) return null;
  return {
    bloodType: loadedLabels[best.picked.index] || `class_${best.picked.index}`,
    bloodTypeConfidence: Number(best.picked.score),
    bloodBox: best.candidate,
    bloodTypeTop: topScores(best.scores, loadedLabels, 3)
  };
}
