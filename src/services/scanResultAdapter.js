import { deriveNrcNumberFromDetections } from '../utils/nrcNumber';

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

const toText = (value) => {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return '';
};

const BURMESE_DIGITS = ['၀', '၁', '၂', '၃', '၄', '၅', '၆', '၇', '၈', '၉'];

const toBurmeseDigits = (value) => {
  const text = toText(value);
  if (!text) return '';
  return text.replace(/\d/g, (digit) => BURMESE_DIGITS[Number(digit)] ?? digit);
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

  const nrcNumberBurmese =
    toText(apiResult.nrcNumberBurmese) ||
    toText(apiResult.nrc_number_burmese) ||
    toBurmeseDigits(nrcNumber) ||
    derivedNrcNumber ||
    '';

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
    nrcNumber: nrcNumber || derivedNrcNumber,
    nrcNumberBurmese,
    confidence,
    bloodTypeConfidence,
    distinctFeature
  };
}
