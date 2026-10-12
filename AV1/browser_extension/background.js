chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "classify_text") {
    fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: request.text })
    })
    .then(response => response.json())
    .then(data => {
      sendResponse({ sentiment: data.sentiment });
    })
    .catch(error => {
      console.error("[Peacemaker Background] Erro ao conectar com o modelo local:", error);
      sendResponse({ sentiment: "positivo" });
    });

    return true;
  }
});