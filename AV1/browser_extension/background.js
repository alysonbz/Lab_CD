// ==========================================
// BACKGROUND.JS - ONNX LOCAL OTIMIZADO PARA EXTENSÃO
// ==========================================
let session = null;
let vocab = null;
const MAX_LENGTH = 100;
let isInitializing = true;
let initPromise = null;

const ABBREVIATIONS_MAP = {
  'pq': 'por que', 'porq': 'porque', 'oq': 'o que', 'obg': 'obrigado',
  'obgd': 'obrigado', 'obgda': 'obrigada', 'vc': 'voce', 'vcs': 'voces',
  'tb': 'tambem', 'tbm': 'tambem', 'blz': 'beleza', 'flw': 'falou',
  'vlw': 'valeu', 'mto': 'muito', 'mta': 'muita', 'mtos': 'muitos',
  'mtas': 'muitas', 'pfv': 'por favor', 'pf': 'por favor', 'kd': 'cade',
  'qdo': 'quando', 'qto': 'quanto', 'hj': 'hoje', 'dps': 'depois',
  'tmj': 'estamos juntos', 'add': 'adicionar', 'att': 'atenciosamente',
  'msg': 'mensagem', 'msgs': 'mensagens', 'qtd': 'quantidade',
  'qtde': 'quantidade', 'prod': 'produto', 'prods': 'produtos'
};

async function initLocalEngine() {
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      console.log("%c[Peacemaker Local] Inicializando motor de IA...", "color: #00bcd4; font-weight: bold;");
      
      if (typeof ort !== 'undefined') {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = false;
        ort.env.wasm.proxy = false;
        
        if (ort.env.wasm.wasmPaths) {
          ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");
        }
      } else {
        throw new Error("A biblioteca ONNX Runtime ('ort') não foi carregada no background.");
      }

      console.log("[Peacemaker Local] Carregando 'vocab.json'...");
      const vocabUrl = chrome.runtime.getURL("vocab.json");
      const vocabResponse = await fetch(vocabUrl);
      if (!vocabResponse.ok) throw new Error("Falha ao carregar vocab.json");
      vocab = await vocabResponse.json();
      console.log(`[Peacemaker Local] Vocabulário carregado! Termos: ${Object.keys(vocab).length}`);

      console.log("[Peacemaker Local] Carregando modelo ONNX ('models/bilstm_model.onnx')...");
      const modelUrl = chrome.runtime.getURL("models/bilstm_model.onnx");
      const externalDataUrl = chrome.runtime.getURL("models/bilstm_model.onnx.data");

      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        externalData: [
          {
            path: "bilstm_model.onnx.data",
            data: externalDataUrl
          }
        ]
      });
      
      isInitializing = false;
      console.log("%c[Peacemaker Local] SUCESSO: Motor ONNX pronto e rodando localmente!", "color: #4caf50; font-weight: bold;");
    } catch (error) {
      isInitializing = false;
      console.error("%c[Peacemaker Local] ERRO CRÍTICO NA INICIALIZAÇÃO:", "color: #f44336; font-weight: bold;", error);
    }
  })();

  return initPromise;
}

initLocalEngine();

function preprocessAndTokenize(text) {
  if (!vocab || typeof text !== 'string') return null;

  let cleaned = text.toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");

  cleaned = cleaned.replace(/o que gostei:/g, 'gostei');
  cleaned = cleaned.replace(/o que nao gostei:/g, 'nao_gostei');
  cleaned = cleaned.replace(/http\S+|www\S+|https\S+/g, '');
  cleaned = cleaned.replace(/\@\w+|\#\w+/g, '');
  cleaned = cleaned.replace(/[^a-z\s]/g, ' ');
  cleaned = cleaned.replace(/\s+/g, ' ').trim();

  let words = cleaned.split(' ');
  let expandedWords = [];

  for (let word of words) {
    if (ABBREVIATIONS_MAP[word]) {
      const expanded = ABBREVIATIONS_MAP[word].split(' ');
      expandedWords.push(...expanded);
    } else if (word.length > 1) {
      expandedWords.push(word);
    }
  }

  let seq = expandedWords.map(w => vocab[w] !== undefined ? vocab[w] : (vocab['<OOV>'] || 1));

  const padToken = vocab['<PAD>'] !== undefined ? vocab['<PAD>'] : 0;
  if (seq.length < MAX_LENGTH) {
    while (seq.length < MAX_LENGTH) {
      seq.push(padToken);
    }
  } else {
    seq = seq.slice(0, MAX_LENGTH);
  }

  return seq;
}

async function runLocalInference(text) {
  if (isInitializing) {
    console.log("[Peacemaker Local] Aguardando o motor terminar de carregar para analisar:", text);
    await initLocalEngine();
  }

  if (!session || !vocab) {
    console.warn("[Peacemaker Local] Motor indisponível. Aplicando fallback.");
    return { sentiment: "positivo", confidence: 0 };
  }

  const seq = preprocessAndTokenize(text);
  if (!seq) return { sentiment: "positivo", confidence: 0 };

  try {
    const inputTensor = new ort.Tensor('int64', BigInt64Array.from(seq.map(v => BigInt(v))), [1, MAX_LENGTH]);
    const inputName = session.inputNames[0];
    
    const feeds = {};
    feeds[inputName] = inputTensor;

    const outputMap = await session.run(feeds);
    const outputName = session.outputNames[0];
    const logits = outputMap[outputName].data;

    let probability = 0;
    let pred = 0;

    if (logits.length > 1) {
      const exp0 = Math.exp(logits[0]);
      const exp1 = Math.exp(logits[1]);
      const sum = exp1 + exp0;
      probability = exp1 / sum;
      pred = probability >= 0.5 ? 1 : 0;
    } else {
      probability = 1 / (1 + Math.exp(-logits[0]));
      pred = probability >= 0.5 ? 1 : 0;
    }

    const sentiment = pred === 1 ? "positivo" : "negativo";
    const confidence = (probability * 100).toFixed(2);

    console.log(`%c[Peacemaker Classificação] Texto: "${text}"`, "color: #ff9800;");
    console.log(`%c  -> Sentimento: ${sentiment.toUpperCase()} | Confiança: ${confidence}%`, "color: #4caf50; font-weight: bold;");

    return { sentiment, confidence };
  } catch (err) {
    console.error("%c[Peacemaker Local] Erro na inferência ONNX:", "color: #f44336; font-weight: bold;", err);
    return { sentiment: "positivo", confidence: 0 };
  }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "classify_text") {
    runLocalInference(request.text)
      .then(result => sendResponse({ sentiment: result.sentiment }))
      .catch(err => {
        console.error("[Peacemaker Background] Erro no listener:", err);
        sendResponse({ sentiment: "positivo" });
      });

    return true;
  }
});