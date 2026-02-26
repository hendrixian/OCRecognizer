export const BURMESE_CLASS_LABELS = Object.freeze([
  '\u1040', '\u1041', '\u1042', '\u1043', '\u1044',
  '\u1045', '\u1046', '\u1047', '\u1048', '\u1049',
  '\u1014\u102d\u102f\u1004\u103a',
  '\u1000', '\u1001', '\u1002', '\u1003', '\u1004', '\u1005', '\u1006', '\u1007',
  '\u100a', '\u100f', '\u1010', '\u1011', '\u1012', '\u1013', '\u1014', '\u1015',
  '\u1016', '\u1017', '\u1018', '\u1019', '\u101a', '\u101b', '\u101c', '\u101d',
  '\u101e', '\u101f', '\u1021', '\u1025'
]);

export const getBurmeseClassLabel = (cls) => {
  if (typeof cls !== 'number' || Number.isNaN(cls)) return '';
  const idx = Math.round(cls);
  const label = BURMESE_CLASS_LABELS[idx];
  return label ?? `${idx}`;
};
