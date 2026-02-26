export const DEFAULT_SCAN_RESULT = {
  nrcNumber: '',
  name: '',
  birthDate: '',
  fatherName: '',
  motherName: '',
  religion: '',
  height: '',
  distinctFeature: '',
  bloodType: '',
  bloodTypeConfidence: 0,
  address: '',
  issueDate: '',
  expiryDate: '',
  confidence: 0
};

import { deriveNrcNumberFromDetections } from '../utils/nrcNumber';

const toText = (value) => {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return '';
};

export function toUiScanResult(apiResult) {
  if (!apiResult || typeof apiResult !== 'object') {
    return { ...DEFAULT_SCAN_RESULT };
  }

  const nrcNumber =
    toText(apiResult.nrcNumber) ||
    toText(apiResult.nrc_number) ||
    toText(apiResult.nrcNumberLatin) ||
    toText(apiResult.nrcNumberBurmese) ||
    toText(apiResult.nrc_number_burmese) ||
    toText(apiResult.rawDigits) ||
    toText(apiResult.raw_digits) ||
    '';

  const derivedNrcNumber = deriveNrcNumberFromDetections(
    apiResult?.boxes || [],
    apiResult?.regionBoxes || apiResult?.areaBoxes || []
  );

  const confidence =
    typeof apiResult.confidence === 'number'
      ? apiResult.confidence
      : DEFAULT_SCAN_RESULT.confidence;

  const bloodTypeConfidence =
    typeof apiResult.bloodTypeConfidence === 'number'
      ? apiResult.bloodTypeConfidence
      : DEFAULT_SCAN_RESULT.bloodTypeConfidence;

  const distinctFeature =
    apiResult.distinctFeature ||
    apiResult.distinct_feature ||
    apiResult.feature ||
    DEFAULT_SCAN_RESULT.distinctFeature;

  return {
    ...DEFAULT_SCAN_RESULT,
    ...apiResult,
    nrcNumber: derivedNrcNumber || nrcNumber,
    confidence,
    bloodTypeConfidence,
    distinctFeature
  };
}
