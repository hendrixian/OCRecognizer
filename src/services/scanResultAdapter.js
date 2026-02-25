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

export function toUiScanResult(apiResult) {
  if (!apiResult || typeof apiResult !== 'object') {
    return { ...DEFAULT_SCAN_RESULT };
  }

  const nrcNumber =
    apiResult.nrcNumber ||
    apiResult.nrc_number ||
    apiResult.nrcNumberLatin ||
    apiResult.nrcNumberBurmese ||
    apiResult.nrc_number_burmese ||
    apiResult.rawDigits ||
    apiResult.raw_digits ||
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
    nrcNumber,
    confidence,
    bloodTypeConfidence,
    distinctFeature
  };
}
