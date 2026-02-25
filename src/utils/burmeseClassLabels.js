export const BURMESE_CLASS_LABELS = Object.freeze([
  '၀', '၁', '၂', '၃', '၄', '၅', '၆', '၇', '၈', '၉',
  'နိုင်',
  'က', 'ခ', 'ဂ', 'ဃ', 'င', 'စ', 'ဆ', 'ဇ', 'ည', 'ဏ', 'တ', 'ထ', 'ဒ', 'ဓ', 'န', 'ပ',
  'ဖ', 'ဗ', 'ဘ', 'မ', 'ယ', 'ရ', 'လ', 'ဝ', 'သ', 'ဟ', 'အ', 'ဥ'
]);

export const getBurmeseClassLabel = (cls) => {
  if (typeof cls !== 'number' || Number.isNaN(cls)) return '';
  const idx = Math.round(cls);
  const label = BURMESE_CLASS_LABELS[idx];
  return label ?? `${idx}`;
};
