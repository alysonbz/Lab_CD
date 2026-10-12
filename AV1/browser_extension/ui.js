// ==========================================
// MÓDULO DE INTERFACE (UI)
// ==========================================
const IMG_CLOSED = typeof browser !== 'undefined' && browser.runtime 
  ? browser.runtime.getURL('assets/peacemaker_closed.png') 
  : 'assets/peacemaker_closed.png';

const IMG_OPEN = typeof browser !== 'undefined' && browser.runtime 
  ? browser.runtime.getURL('assets/peacemaker_open.png') 
  : 'assets/peacemaker_open.png';

const LOGO_IMG_HORIZONTAL = typeof browser !== 'undefined' && browser.runtime 
  ? browser.runtime.getURL('assets/peacemaker_logo_horizontal.png') 
  : 'assets/peacemaker_logo_horizontal.png';

let activeAnimationCleanup = null;
let currentInsistenceLevel = 0;

function generateSingleSuggestion(context) {
  const pool = constructiveFragments[context] || constructiveFragments[PeacemakerContext.SOCIAL];
  return pool[Math.floor(Math.random() * pool.length)];
}

function generateSuggestionOptions(context) {
  const suggestions = new Set();
  while (suggestions.size < 3) {
    suggestions.add(generateSingleSuggestion(context));
  }
  return Array.from(suggestions);
}

function showPeacemakerDialogueOverlay(messageText, inputElement, onProceed, getContextTypeFn) {
  const existingOverlay = document.getElementById("peacemaker-rpg-overlay");
  if (existingOverlay) {
    if (activeAnimationCleanup) activeAnimationCleanup();
    existingOverlay.remove();
  }

  const context = getContextTypeFn();
  const fullMessage = getPersistentMessage(context, currentInsistenceLevel);
  const suggestions = generateSuggestionOptions(context);

  const rawSentences = fullMessage.match(/[^.!?]+[.!?]+/g) || [fullMessage];
  const sentences = rawSentences.map(s => s.trim());

  const overlay = document.createElement("div");
  overlay.id = "peacemaker-rpg-overlay";
  overlay.style.cssText = `
    position: fixed; top: 25px; right: 25px; z-index: 999999;
    display: flex; align-items: flex-start; gap: 15px;
    font-family: 'Comic Sans MS', 'Segoe UI', Arial, sans-serif;
    max-width: 480px; pointer-events: auto;
  `;

  const dialogueContainer = document.createElement("div");
  dialogueContainer.style.cssText = `
    display: flex; flex-direction: column; gap: 10px; width: 100%; align-items: flex-end;
  `;

  const speechBubble = document.createElement("div");
  speechBubble.style.cssText = `
    background: #ffffff; color: #111111; padding: 15px 18px; border-radius: 18px;
    border: 3px solid #000000; box-shadow: 4px 4px 0px rgba(0,0,0,0.8);
    font-size: 14.5px; line-height: 1.4; min-height: 50px; width: 100%;
    min-width: 220px; box-sizing: border-box;
    word-break: break-word; position: relative; font-weight: bold;
  `;

  const tail = document.createElement("div");
  tail.style.cssText = `
    position: absolute; right: -12px; top: 22px; width: 0; height: 0;
    border-top: 8px solid transparent; border-bottom: 8px solid transparent;
    border-left: 12px solid #ffffff; filter: drop-shadow(2px 1px 0px #000);
  `;
  speechBubble.appendChild(tail);

  const logoTitle = document.createElement("img");
  logoTitle.src = LOGO_IMG_HORIZONTAL;
  logoTitle.alt = "Peacemaker-Reborn";
  logoTitle.style.cssText = `
    position: absolute; top: 10px; right: 15px; width: 150px; height: auto;
    display: block; filter: drop-shadow(1px 1px 0px #000); z-index: 2;
  `;

  const textElement = document.createElement("div");
  textElement.style.cssText = `
    color: #222222; font-weight: normal; white-space: pre-wrap; margin-top: 35px;
  `;
  
  speechBubble.appendChild(logoTitle);
  speechBubble.appendChild(textElement);
  dialogueContainer.appendChild(speechBubble);

  const optionsContainer = document.createElement("div");
  optionsContainer.style.cssText = `
    display: none; flex-direction: column; gap: 8px; background: #fffecb;
    padding: 14px; border-radius: 16px; border: 3px solid #000000; width: 100%;
    box-sizing: border-box; box-shadow: 4px 4px 0px rgba(0,0,0,0.8);
  `;

  const optionsHeader = document.createElement("div");
  optionsHeader.style.cssText = "color: #d9534f; font-size: 13px; font-weight: 900; margin-bottom: 4px; text-transform: uppercase;";
  optionsHeader.innerText = currentInsistenceLevel === 0 ? "💬 O que prefere dizer?" : `🚨 NÍVEL DE INSISTÊNCIA ${currentInsistenceLevel + 1}: ESCOLHA A PAZ!`;
  optionsContainer.appendChild(optionsHeader);

  const avatarWrapper = document.createElement("div");
  avatarWrapper.style.cssText = `
    width: 100px; height: 100px; border-radius: 50%; overflow: hidden;
    border: 3px solid #000; box-shadow: 0 0 15px rgba(255, 75, 75, 0.7);
    background: #111; flex-shrink: 0; position: relative;
  `;

  const avatarImg = document.createElement("img");
  avatarImg.src = IMG_CLOSED;
  avatarImg.alt = "Peacemaker";
  avatarImg.style.cssText = `
    width: 100%; height: 100%; object-fit: cover; object-position: center 15%; transform: scale(1.65); transform-origin: top center;
  `;
  avatarWrapper.appendChild(avatarImg);

  overlay.appendChild(dialogueContainer);
  overlay.appendChild(avatarWrapper);
  document.body.appendChild(overlay);

  let currentSentenceIdx = 0;
  let activeMouthTimeout = null;
  let cancelTypingFlag = false;

  function triggerMouthMovement(word) {
    const cleanWord = word.replace(/[^a-zA-ZáàâãéèêíïóôõöúçÑñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇ]/g, "");
    if (!cleanWord.length) return;

    let cycles = 1;
    if (cleanWord.length >= 8) cycles = 3;
    else if (cleanWord.length >= 4) cycles = 2;

    const speedMs = 70;
    let currentCycle = 0;

    function step() {
      if (cancelTypingFlag || currentCycle >= cycles) {
        avatarImg.src = IMG_CLOSED;
        return;
      }

      avatarImg.src = IMG_OPEN;
      activeMouthTimeout = setTimeout(() => {
        avatarImg.src = IMG_CLOSED;
        activeMouthTimeout = setTimeout(() => {
          currentCycle++;
          step();
        }, speedMs);
      }, speedMs);
    }

    step();
  }

  async function playNextSentence() {
    if (cancelTypingFlag) return;

    if (currentSentenceIdx >= sentences.length) {
      if (avatarImg) avatarImg.src = IMG_CLOSED;
      optionsContainer.style.display = "flex";
      dialogueContainer.appendChild(optionsContainer);
      return;
    }

    const currentText = sentences[currentSentenceIdx];
    textElement.textContent = '';

    const words = currentText.split(/(\s+)/);
    const charSpeedMs = 40;
    const totalAnimationDurationMs = currentText.length * charSpeedMs;

    const ttsPromise = speakPeacemakerText(currentText, totalAnimationDurationMs);

    for (const word of words) {
      if (cancelTypingFlag) break;

      if (word.trim().length > 0) {
        triggerMouthMovement(word);
      }

      for (let i = 0; i < word.length; i++) {
        if (cancelTypingFlag) break;
        textElement.textContent += word.charAt(i);
        await new Promise(r => setTimeout(r, charSpeedMs));
      }
    }

    textElement.textContent = currentText;
    await ttsPromise;

    if (cancelTypingFlag) return;

    avatarImg.src = IMG_CLOSED;

    setTimeout(() => {
      if (!cancelTypingFlag) {
        currentSentenceIdx++;
        playNextSentence();
      }
    }, 250);
  }

  suggestions.forEach((suggestion) => {
    const btn = document.createElement("button");
    btn.style.cssText = `
      background: #ffffff; color: #111111; border: 2px solid #000000;
      padding: 10px 12px; border-radius: 10px; cursor: pointer; text-align: left;
      font-size: 13px; transition: all 0.2s; font-family: 'Comic Sans MS', 'Segoe UI', Arial, sans-serif;
      box-shadow: 2px 2px 0px rgba(0,0,0,0.5); font-weight: normal;
    `;
    btn.innerHTML = `<strong>▸ Falar:</strong> "${suggestion}"`;
    
    btn.onmouseover = () => {
      btn.style.background = "#e6f2ff";
      btn.style.transform = "translateY(-1px)";
    };
    btn.onmouseout = () => {
      btn.style.background = "#ffffff";
      btn.style.transform = "translateY(0)";
    };

    btn.onclick = () => {
      if (activeAnimationCleanup) activeAnimationCleanup();
      currentInsistenceLevel = 0;
      if (inputElement && inputElement.tagName) {
        if (inputElement.isContentEditable) {
          inputElement.innerText = suggestion;
        } else if (inputElement.value !== undefined) {
          inputElement.value = suggestion;
          inputElement.dispatchEvent(new Event('input', { bubbles: true }));
        }
      }
      overlay.remove();
    };

    optionsContainer.appendChild(btn);
  });

  const actionsRow = document.createElement("div");
  actionsRow.style.cssText = "display: flex; gap: 8px; margin-top: 6px;";

  const btnEdit = document.createElement("button");
  btnEdit.style.cssText = "flex: 1; background: #ffcc00; color: #000; border: 2px solid #000; padding: 8px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: bold; box-shadow: 2px 2px 0px #000;";
  btnEdit.innerText = "Mudar de Ideia";
  btnEdit.onclick = () => {
    if (activeAnimationCleanup) activeAnimationCleanup();
    currentInsistenceLevel = 0;
    overlay.remove();
  };

  const btnForce = document.createElement("button");
  const forceBg = currentInsistenceLevel >= 3 ? "#8b0000" : "#ff4b4b";
  btnForce.style.cssText = `flex: 1; background: ${forceBg}; color: #fff; border: 2px solid #000; padding: 8px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: bold; box-shadow: 2px 2px 0px #000;`;
  btnForce.innerText = currentInsistenceLevel >= 3 ? "Enviar Assim Mesmo (Aumentar Caos)" : "Insistir no Envio";
  
  btnForce.onclick = () => {
    if (activeAnimationCleanup) activeAnimationCleanup();
    
    if (PERSISTENCE_CONFIG.enabled && currentInsistenceLevel < PERSISTENCE_CONFIG.maxLevels - 1) {
      currentInsistenceLevel++;
      overlay.remove();
      showPeacemakerDialogueOverlay(messageText, inputElement, onProceed, getContextTypeFn);
    } else {
      currentInsistenceLevel = 0;
      overlay.remove();
      onProceed();
    }
  };

  actionsRow.appendChild(btnEdit);
  actionsRow.appendChild(btnForce);
  optionsContainer.appendChild(actionsRow);

  playNextSentence();

  activeAnimationCleanup = () => {
    cancelTypingFlag = true;
    if (activeMouthTimeout) clearTimeout(activeMouthTimeout);
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  };
}