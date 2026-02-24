export const DEFAULT_SCAN_RESULT = {
  nrcNumber: '',
  bloodType: '',
  bloodTypeConfidence: 0,
  bloodTypeTop: [],
  name: '',
  birthDate: '',
  fatherName: '',
  motherName: '',
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

  const bloodType =
    apiResult.bloodType ||
    apiResult.blood_type ||
    '';

  const bloodTypeConfidence =
    typeof apiResult.bloodTypeConfidence === 'number'
      ? apiResult.bloodTypeConfidence
      : typeof apiResult.blood_type_confidence === 'number'
        ? apiResult.blood_type_confidence
        : DEFAULT_SCAN_RESULT.bloodTypeConfidence;

  return {
    ...DEFAULT_SCAN_RESULT,
    ...apiResult,
    nrcNumber,
    confidence,
    bloodType,
    bloodTypeConfidence
  };
}
