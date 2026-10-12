// ==========================================
// MÓDULO DE TTS (SINTETIZADOR DE VOZ) - CORRIGIDO
// ==========================================
function speakPeacemakerText(text, targetDurationMs) {
  return new Promise(async (resolve) => {
    if (!('speechSynthesis' in window)) {
      resolve();
      return;
    }

    window.speechSynthesis.cancel();

    let voices = window.speechSynthesis.getVoices();
    if (voices.length === 0) {
      voices = await new Promise((resolveVoices) => {
        window.speechSynthesis.onvoiceschanged = () => {
          resolveVoices(window.speechSynthesis.getVoices());
        };
        setTimeout(() => {
          resolveVoices(window.speechSynthesis.getVoices());
        }, 150);
      });
    }

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