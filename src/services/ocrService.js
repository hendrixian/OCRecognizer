export async function scanNrcFromDataUrl(imageDataUrl) {
  if (!imageDataUrl) {
    throw new Error('No image selected');
  }

  if (!window?.electronAPI?.scanNrc) {
    throw new Error('OCR service is not available');
  }

  const result = await window.electronAPI.scanNrc({ imageDataUrl });

  if (!result?.ok) {
    throw new Error(result?.error || 'Scan failed');
  }

  return result.data;
}
