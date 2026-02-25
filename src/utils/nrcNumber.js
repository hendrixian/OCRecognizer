import { getBurmeseClassLabel } from './burmeseClassLabels';

const normalizeLabel = (value) =>
  String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '');

const isNrcRegionLabel = (label) => {
  const normalized = normalizeLabel(label);
  if (!normalized) return false;
  if (normalized === 'nrcnumber') return true;
  if (normalized === 'nrcnum') return true;
  if (normalized === 'nrcno') return true;
  return normalized.startsWith('nrcnumber');
};

const pickNrcRegion = (regions) => {
  if (!Array.isArray(regions) || regions.length === 0) return null;
  const candidates = regions.filter((region) => isNrcRegionLabel(region?.label));
  if (candidates.length === 0) return null;
  return candidates.reduce((best, current) => {
    const bestConf = typeof best?.conf === 'number' ? best.conf : 0;
    const currentConf = typeof current?.conf === 'number' ? current.conf : 0;
    return currentConf > bestConf ? current : best;
  }, candidates[0]);
};

const isCenterInside = (box, region) => {
  if (!box || !region) return false;
  const x1 = typeof box.x1 === 'number' ? box.x1 : null;
  const y1 = typeof box.y1 === 'number' ? box.y1 : null;
  const x2 = typeof box.x2 === 'number' ? box.x2 : null;
  const y2 = typeof box.y2 === 'number' ? box.y2 : null;
  if (x1 === null || y1 === null || x2 === null || y2 === null) return false;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  return cx >= region.x1 && cx <= region.x2 && cy >= region.y1 && cy <= region.y2;
};

export const deriveNrcNumberFromDetections = (boxes = [], regions = []) => {
  if (!Array.isArray(boxes) || boxes.length === 0) return '';

  const region = pickNrcRegion(regions);
  const candidates = region ? boxes.filter((box) => isCenterInside(box, region)) : boxes;
  if (!candidates || candidates.length === 0) return '';

  const ordered = [...candidates].sort((a, b) => {
    const ax = typeof a.x1 === 'number' ? a.x1 : 0;
    const bx = typeof b.x1 === 'number' ? b.x1 : 0;
    if (ax !== bx) return ax - bx;
    const ay = typeof a.y1 === 'number' ? a.y1 : 0;
    const by = typeof b.y1 === 'number' ? b.y1 : 0;
    return ay - by;
  });

  const labels = ordered
    .map((box) => getBurmeseClassLabel(box?.cls))
    .filter((label) => typeof label === 'string' && label.length > 0);

  return labels.join('');
};
