// ==========================================
// 1. MÓDULO DE MENSAGENS DO PEACEMAKER
// ==========================================
const PeacemakerContext = {
  SEARCH: "search",
  SOCIAL: "social"
};

const peacemakerFragments = {
  search: {
    intros: [
      "Olha só, campeão...",
      "Nossa, que pesquisa iluminada...",
      "Amigo, escuta aqui...",
      "Olha o tamanho dessa energia negativa..."
    ],
    middles: [
      "pesquisar isso com essa raiva toda não vai trazer a paz mundial.",
      "usar o Google para descarregar o estresse é um desperdício do potencial humano.",
      "parece até que você está querendo declarar guerra contra o teclado.",
      "escrever termos tão carregados assim só afasta a harmonia do universo."
    ],
    endings: [
      "O Eagly e eu achamos que você pode digitar algo bem mais construtivo, né?",
      "Que tal tentar termos pacíficos para o bem da humanidade?",
      "Vamos respirar fundo, endireitar a postura e procurar algo que ajude a construir a paz?",
      "Tenta reformular essa busca com mais amor no coração e menos drama."
    ]
  },
  social: {
    intros: [
      "Nossa, que comentário adorável...",
      "Caramba, quanta energia pesada em uma frase só...",
      "Olha, meu caro...",
      "Se o objetivo era espalhar discórdia..."
    ],
    middles: [
      "parece que o único objetivo era fazer todo mundo revirar os olhos.",
      "esse texto está testando seriamente o meu limite de paciência com a diplomacia.",
      "parece até que você esqueceu completamente onde fica o botão de apagar.",
      "escrever algo assim na internet é quase pedir um abraço virtual ou um sermão."
    ],
    endings: [
      "Que tal espalhar amor de verdade, igualzinho o Eagly faz?",
      "O Comitê de Paz adoraria que você reescrevesse isso com um sorriso no rosto.",
      "Que tal um texto mais fofo e acolhedor para a nossa comunidade?",
      "Vamos fazer a internet um lugar melhor e digitar algo construtivo?"
    ]
  }
};

function generatePeacemakerMessage(context) {
  const pool = peacemakerFragments[context] || peacemakerFragments[PeacemakerContext.SOCIAL];
  const intro = pool.intros[Math.floor(Math.random() * pool.intros.length)];
  const middle = pool.middles[Math.floor(Math.random() * pool.middles.length)];
  const ending = pool.endings[Math.floor(Math.random() * pool.endings.length)];
  return `${intro} ${middle} ${ending}`;
}


// ==========================================
// 2. MÓDULO DE SUGESTÕES ALEATÓRIAS E DIVERTIDAS
// ==========================================
const constructiveFragments = {
  search: [
    "onde achar gatos fofinhos para adotar?",
    "melhores sabores de sorvete artesanal na cidade",
    "como ensinar meu cachorro a dar a pata",
    "receita de bolo de chocolate super fácil e rápida",
    "lugares tranquilos para ver o pôr do sol",
    "curiosidades fascinantes sobre águias e aves de rapina",
    "filmes de comédia leve para rir hoje"
  ],
  social: [
    "Acho que todo mundo merece comer um bom hambúrguer hoje! 🍔",
    "Hoje o dia está perfeito para tomar um açaí bem caprichado!",
    "Vocês já viram como o Eagly é uma belezinha? 🦅✨",
    "Mandando muita energia positiva e abraços virtuais para todo mundo!",
    "Vamos focar nas coisas boas da vida e espalhar sorrisos!",
    "Alguém recomenda uma série legal para maratonar no fim de semana?",
    "Paz, amor e muita música boa para todos!"
  ]
};

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


// ==========================================
// 3. MÓDULO DE TTS DINÂMICO E VOZ GRAVE
// ==========================================
function speakPeacemakerText(text, targetDurationMs) {
  if (!('speechSynthesis' in window)) return;

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'pt-BR';
  utterance.pitch = 0.5;

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

  window.speechSynthesis.speak(utterance);
}


// ==========================================
// 4. MÓDULO DE ANIMAÇÃO E INTERFACE (COMIC RPG DIALOGUE)
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

function showPeacemakerDialogueOverlay(messageText, inputElement, onProceed) {
  const existingOverlay = document.getElementById("peacemaker-rpg-overlay");
  if (existingOverlay) {
    if (activeAnimationCleanup) activeAnimationCleanup();
    existingOverlay.remove();
  }

  const context = getContextType();
  const fullMessage = generatePeacemakerMessage(context);
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
    position: absolute;
    top: 10px;
    right: 15px;
    width: 150px;
    height: auto;
    display: block;
    filter: drop-shadow(1px 1px 0px #000);
    z-index: 2;
  `;

  const textElement = document.createElement("div");
  textElement.style.cssText = `
    color: #222222;
    font-weight: normal;
    white-space: pre-wrap;
    margin-top: 35px;
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
  optionsHeader.innerText = "💬 O que prefere dizer?";
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
  let talkInterval = null;
  let typeInterval = null;

  function playNextSentence() {
    if (currentSentenceIdx >= sentences.length) {
      if (avatarImg) avatarImg.src = IMG_CLOSED;
      optionsContainer.style.display = "flex";
      dialogueContainer.appendChild(optionsContainer);
      return;
    }

    const currentText = sentences[currentSentenceIdx];
    textElement.textContent = '';
    let charIndex = 0;
    let isOpen = false;

    const typeSpeedMs = 30;
    const totalAnimationDurationMs = currentText.length * typeSpeedMs;

    speakPeacemakerText(currentText, totalAnimationDurationMs);

    if (talkInterval) clearInterval(talkInterval);
    talkInterval = setInterval(() => {
      isOpen = !isOpen;
      avatarImg.src = isOpen ? IMG_OPEN : IMG_CLOSED;
    }, 180);

    if (typeInterval) clearInterval(typeInterval);
    typeInterval = setInterval(() => {
      if (charIndex < currentText.length) {
        textElement.textContent += currentText.charAt(charIndex);
        charIndex++;
      } else {
        clearInterval(typeInterval);
        clearInterval(talkInterval);
        avatarImg.src = IMG_CLOSED;

        setTimeout(() => {
          currentSentenceIdx++;
          playNextSentence();
        }, 900);
      }
    }, typeSpeedMs);
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
      if (inputElement.isContentEditable) {
        inputElement.innerText = suggestion;
      } else {
        inputElement.value = suggestion;
        inputElement.dispatchEvent(new Event('input', { bubbles: true }));
      }
      overlay.remove();
    };

    optionsContainer.appendChild(btn);
  });

  const actionsRow = document.createElement("div");
  actionsRow.style.cssText = "display: flex; gap: 8px; margin-top: 6px;";

  const btnEdit = document.createElement("button");
  btnEdit.style.cssText = "flex: 1; background: #ffcc00; color: #000; border: 2px solid #000; padding: 8px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: bold; box-shadow: 2px 2px 0px #000;";
  btnEdit.innerText = "Escrever Outra";
  btnEdit.onclick = () => {
    if (activeAnimationCleanup) activeAnimationCleanup();
    overlay.remove();
  };

  const btnForce = document.createElement("button");
  btnForce.style.cssText = "flex: 1; background: #ff4b4b; color: #fff; border: 2px solid #000; padding: 8px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: bold; box-shadow: 2px 2px 0px #000;";
  btnForce.innerText = "Enviar Assim Mesmo";
  btnForce.onclick = () => {
    if (activeAnimationCleanup) activeAnimationCleanup();
    overlay.remove();
    onProceed();
  };

  actionsRow.appendChild(btnEdit);
  actionsRow.appendChild(btnForce);
  optionsContainer.appendChild(actionsRow);

  playNextSentence();

  activeAnimationCleanup = () => {
    if (talkInterval) clearInterval(talkInterval);
    if (typeInterval) clearInterval(typeInterval);
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  };
}


// ==========================================
// 5. LÓGICA PRINCIPAL DE INTERCEPÇÃO
// ==========================================
function getContextType() {
  const url = window.location.hostname;
  if (url.includes("google")) {
    return PeacemakerContext.SEARCH;
  }
  return PeacemakerContext.SOCIAL; 
}

async function classifyText(text) {
  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    const data = await response.json();
    return data.sentiment; 
  } catch (error) {
    console.error("[Peacemaker-Reborn] Erro ao conectar com o modelo local:", error);
    return "positivo"; 
  }
}

let isExplicitlyAllowed = false;

function handleEventIntercept(event, inputElement, submitAction) {
  if (isExplicitlyAllowed) {
    isExplicitlyAllowed = false;
    return;
  }

  const messageText = inputElement.value || inputElement.innerText;
  if (!messageText || messageText.trim().length < 3) return;

  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation();

  classifyText(messageText).then((sentiment) => {
    if (sentiment === "negativo") {
      showPeacemakerDialogueOverlay(messageText, inputElement, () => {
        isExplicitlyAllowed = true;
        submitAction();
      });
    } else {
      isExplicitlyAllowed = true;
      submitAction();
    }
  });
}

document.addEventListener("submit", (event) => {
  const form = event.target;
  const inputElement = form.querySelector('textarea, input[type="text"], input[type="search"], div[contenteditable="true"]');
  if (!inputElement) return;

  handleEventIntercept(event, inputElement, () => {
    HTMLFormElement.prototype.submit.call(form);
  });
}, true);

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    const target = event.target;
    const isInputField = target.matches('textarea, input[type="text"], input[type="search"], div[contenteditable="true"]');
    if (!isInputField) return;

    handleEventIntercept(event, target, () => {
      const form = target.closest("form");
      if (form) {
        HTMLFormElement.prototype.submit.call(form);
      } else {
        const searchBtn = document.querySelector('button.HZVG1b') || document.querySelector('button[type="submit"]');
        if (searchBtn) searchBtn.click();
      }
    });
  }
}, true);