// ==========================================
// MÓDULO PRINCIPAL DE CONTEÚDO (CONTENT.JS)
// ==========================================
function getContextType() {
  const url = window.location.hostname;
  if (url.includes("google")) {
    return PeacemakerContext.SEARCH;
  }
  return PeacemakerContext.SOCIAL; 
}

async function classifyText(text) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ action: "classify_text", text: text }, (response) => {
      if (chrome.runtime.lastError || !response) {
        console.error("[Peacemaker-Reborn] Erro na comunicação com o background:", chrome.runtime.lastError);
        resolve("positivo"); 
      } else {
        resolve(response.sentiment);
      }
    });
  });
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
      }, getContextType);
    } else {
      isExplicitlyAllowed = true;
      submitAction();
    }
  });
}

function executeStandardSubmit(targetElement) {
  const form = targetElement.closest("form");
  
  if (form) {
    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
    } else {
      HTMLFormElement.prototype.submit.call(form);
    }
  } else {
    const searchBtn = document.querySelector('button.HZVG1b') || 
                      document.querySelector('button[type="submit"]') || 
                      document.querySelector('form[role="search"] button') ||
                      document.querySelector('div[role="button"]');
    if (searchBtn) {
      searchBtn.click();
    } else {
      const enterEvent = new KeyboardEvent('keydown', {
        key: 'Enter',
        code: 'Enter',
        keyCode: 13,
        which: 13,
        bubbles: true,
        cancelable: true
      });
      targetElement.dispatchEvent(enterEvent);
    }
  }
}

document.addEventListener("submit", (event) => {
  if (isExplicitlyAllowed) {
    isExplicitlyAllowed = false;
    return;
  }

  const form = event.target;
  const inputElement = form.querySelector('textarea, input[type="text"], input[type="search"], div[contenteditable="true"]');
  if (!inputElement) return;

  handleEventIntercept(event, inputElement, () => {
    executeStandardSubmit(inputElement);
  });
}, true);

document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    if (isExplicitlyAllowed) {
      isExplicitlyAllowed = false;
      return;
    }

    const target = event.target;
    const isInputField = target.matches('textarea, input[type="text"], input[type="search"], div[contenteditable="true"]');
    if (!isInputField) return;

    handleEventIntercept(event, target, () => {
      executeStandardSubmit(target);
    });
  }
}, true);

function checkAddressBarSearchInterception() {
  const urlParams = new URLSearchParams(window.location.search);
  const queryParam = urlParams.get('q') || urlParams.get('query') || urlParams.get('p');
  
  if (!queryParam || queryParam.trim().length < 3) return;

  const sessionKey = 'peacemaker_checked_address_bar_' + queryParam;
  if (sessionStorage.getItem(sessionKey)) return;
  sessionStorage.setItem(sessionKey, 'true');

  classifyText(queryParam).then((sentiment) => {
    if (sentiment === "negativo") {
      const searchInput = document.querySelector('textarea, input[type="text"], input[type="search"], div[contenteditable="true"]');
      
      showPeacemakerDialogueOverlay(queryParam, searchInput, () => {
        sessionStorage.removeItem(sessionKey);
      }, getContextType);
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', checkAddressBarSearchInterception);
} else {
  checkAddressBarSearchInterception();
}