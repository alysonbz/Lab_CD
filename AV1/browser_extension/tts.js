// ==========================================
// MÓDULO DE TTS (SINTETIZADOR DE VOZ)
// ==========================================
function speakPeacemakerText(text, targetDurationMs) {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) {
      resolve();
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'pt-BR';
    utterance.pitch = 0.45;

    if (targetDurationMs && text.length > 0) {
      const baseDurationMs = (text.length / 13) * 1000;
      let calculatedRate = baseDurationMs / targetDurationMs;
      utterance.rate = Math.min(Math.max(calculatedRate, 0.7), 1.6);
    } else {
      utterance.rate = 0.95;
    }

    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => 
      v.lang.startsWith('pt') && 
      (v.name.toLowerCase().includes('natural') || 
       v.name.toLowerCase().includes('google') || 
       v.name.toLowerCase().includes('daniel') || 
       v.name.toLowerCase().includes('ricardo'))
    ) || voices.find(v => v.lang.startsWith('pt'));

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();

    window.speechSynthesis.speak(utterance);
  });
}