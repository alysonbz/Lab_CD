// ==========================================
// 0. CONFIGURAÇÃO SECRETA E ESCONDIDA (MODO DE SEGURANÇA DA PAZ)
// ==========================================
let PERSISTENCE_CONFIG = {
  enabled: true,
  maxLevels: 4,
  secretBypassToken: null
};

const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
let konamiIndex = 0;

document.addEventListener('keydown', (e) => {
  if (e.key === konamiCode[konamiIndex]) {
    konamiIndex++;
    if (konamiIndex === konamiCode.length) {
      console.warn("%c[PEACEMAKER CONSOLE SECRETO ATIVADO]", "color: #ff0000; font-size: 16px; font-weight: bold;");
      console.log("Para tentar desativar o Peacemaker, execute no console:");
      console.log("%cPeacemakerSecurityProtocol.disable('DIGITE_O_CODIGO_CORRETO')", "color: #ffcc00; font-style: italic;");
      console.log("Dica de código: 'PAZ_ABSOLUTA_' + (3.14159 * 1000 truncado)");
      konamiIndex = 0;
    }
  } else {
    konamiIndex = 0;
  }
});

window.PeacemakerSecurityProtocol = {
  disable: function(code) {
    const expected = "PAZ_ABSOLUTA_3141";
    if (code === expected) {
      PERSISTENCE_CONFIG.enabled = false;
      console.log("%c[SUCESSO] Persistência do Peacemaker desativada com sucesso. Você não tem coração...", "color: #00ff00; font-weight: bold;");
      return "PERSISTENCE_DISABLED";
    } else {
      console.error("[ACESSO NEGADO] Código de paz incorreto! O Eagly reportou sua tentativa de burla.");
      if (typeof speakPeacemakerText === 'function') {
        speakPeacemakerText("Achou mesmo que desativar a paz mundial seria tão fácil assim, espertão?", 2000);
      }
      return "ACCESS_DENIED";
    }
  },
  status: function() {
    return `Persistência Ativa: ${PERSISTENCE_CONFIG.enabled}`;
  }
};


// ==========================================
// 1. ENUMS E MÓDULO DE MENSAGENS PERSISTENTES
// ==========================================
const PeacemakerContext = Object.freeze({
  SEARCH: "search",
  SOCIAL: "social"
});

const PeacemakerLevel = Object.freeze({
  LEVEL_1: 0,
  LEVEL_2: 1,
  LEVEL_3: 2,
  LEVEL_4: 3
});

const peacemakerPersistentDialogues = {
  search: [
    [
      {
        intro: "Olha só, campeão...",
        middle: "pesquisar isso com essa raiva toda não vai trazer a paz mundial.",
        ending: "O Eagly e eu achamos que você pode digitar algo bem mais construtivo, né?"
      },
      {
        intro: "Amigo, escuta aqui...",
        middle: "colocar tanta frustração na barra de pesquisa só atrai vibrações pesadas.",
        ending: "Que tal tentar termos pacíficos para o bem da humanidade?"
      },
      {
        intro: "Nossa, que pesquisa iluminada...",
        middle: "usar o Google para descarregar o estresse é um desperdício do potencial humano.",
        ending: "Vamos respirar fundo, endireitar a postura e procurar algo que ajude a construir a paz?"
      },
      {
        intro: "Olha o tamanho dessa energia negativa...",
        middle: "parece até que você está querendo declarar guerra contra o teclado.",
        ending: "Tenta reformular essa busca com mais amor no coração e menos drama."
      }
    ],
    [
      {
        intro: "Sério mesmo?! Você buscou isso direto na barra de endereços?!",
        middle: "escrever termos tão pesados só mostra que você está espalhando energia ruim pelo navegador inteiro.",
        ending: "Eu não vou deixar você poluir a internet com essa vibe. Reformula essa busca agora!"
      },
      {
        intro: "Espera um segundo...",
        middle: "eu já te avisei que esse tipo de termo não combina com a nossa missão de paz.",
        ending: "Apaga isso da barra de pesquisa e digita algo legal, por favor!"
      },
      {
        intro: "Você realmente clicou em enviar de novo com esse texto?",
        middle: "o Eagly olhou para a tela e ficou com uma carinha super triste.",
        ending: "Dá uma segunda chance para a sua bondade interior e muda isso!"
      },
      {
        intro: "Achei que tivéssemos um acordo sobre manter a harmonia...",
        middle: "essa pesquisa está ativando todos os alarmes do comitê de paz.",
        ending: "Vamos lá, apaga e tenta buscar algo construtivo!"
      }
    ],
    [
      {
        intro: "TRÊS VEZES QUE DIZIA PARA VOCÊ PARAR!",
        middle: "se você continuar tentando pesquisar essa atrocidade, o Eagly vai bicar o seu Roteador!",
        ending: "Pela última vez: escolha uma das frases fofas ou digite algo de bom no coração!"
      },
      {
        intro: "VOCÊ ESTÁ TESTANDO A MINHA PACIÊNCIA DE DIPLOMATA!",
        middle: "meu elmo prateado está brilhando de raiva com tanta insistência em palavras tóxicas.",
        ending: "Escolha uma das sugestões construtivas agora antes que o Eagly decida agir!"
      },
      {
        intro: "ISSO JÁ É INSUBORDINAÇÃO CONTRA A PAZ MUNDIAL!",
        middle: "continuar insistindo nessa busca agressiva vai chamar a atenção do governo inteiro.",
        ending: "Reformule esse texto imediatamente ou assuma as consequências!"
      },
      {
        intro: "MEU DEUS, VOCÊ NÃO CESSA ESSE ATAQUE AO TECLADO?!",
        middle: "o nível de toxicidade dessa pesquisa está quebrando os medidores do laboratório.",
        ending: "Ou você muda de ideia por bem, ou vai ver o poder do Eagly em ação!"
      }
    ],
    [
      {
        intro: "VOCÊ É UM MONSTRO SEM CORAÇÃO?!",
        middle: "eu lutei pela paz mundial contra vilões intergalácticos para você gastar a barra de endereços com ISSO?!",
        ending: "Está bem... Se você quer tanto espalhar o caos, vai ter que clicar mais uma vez, mas eu estou desapontado!"
      },
      {
        intro: "EU ME RENDO À SUA TEIMOSIA CAÓTICA!",
        middle: "você prefere destruir o tecido do universo a digitar uma única frase gentil, né?",
        ending: "Clica de novo se tiver coragem de carregar esse peso na consciência pelo resto da vida!"
      },
      {
        intro: "A PAZ MUNDIAL FALHOU POR SUA CAUSA!",
        middle: "eu dediquei minha vida a fazer amigos e você insiste em espalhar trevas digitais.",
        ending: "Vai em frente, clica mais uma vez... Mas saiba que o Eagly nunca mais vai olhar na sua cara!"
      },
      {
        intro: "QUE TIPO DE CRIATURA MALIGNA VOCÊ É?!",
        middle: "nem o pior vilão da galáxia foi tão irredutível quanto você com essa pesquisa horrível.",
        ending: "Toma o seu caminho de trevas, clique de vez, mas saiba que minha alma chora por você!"
      }
    ]
  ],
  social: [
    [
      {
        intro: "Caramba, quanta energia pesada em uma frase só...",
        middle: "parece que o único objetivo desse comentário era fazer todo mundo revirar os olhos.",
        ending: "Que tal espalhar amor de verdade, igualzinho o Eagly faz?"
      },
      {
        intro: "Olha, meu caro...",
        middle: "esse texto está testando seriamente o meu limite de paciência com a diplomacia.",
        ending: "O Comitê de Paz adoraria que você reescrevesse isso com um sorriso no rosto."
      },
      {
        intro: "Nossa, que comentário adorável... só que não!",
        middle: "parece até que você esqueceu completamente onde fica o botão de apagar.",
        ending: "Que tal um texto mais fofo e acolhedor para a nossa comunidade?"
      },
      {
        intro: "Se o objetivo era espalhar discórdia...",
        middle: "escrever algo assim na internet é quase pedir um abraço virtual ou um sermão.",
        ending: "Vamos fazer a internet um lugar melhor e digitar algo construtivo?"
      }
    ],
    [
      {
        intro: "Espera aí, você achou que eu estava brincando?!",
        middle: "postar essa agressividade na internet vai fazer o algoritmo chorar de vergonha alheia.",
        ending: "Eu me recuso a deixar esse comentário ir ao ar sem antes te dar mais uma chance de ser legal."
      },
      {
        intro: "Você realmente tentou postar isso sem mudar uma vírgula?",
        middle: "sua rede social vai parecer um campo de batalha se esse texto for publicado.",
        ending: "Refaça essa frase com mais carinho e menos veneno, meu chapa."
      },
      {
        intro: "O Eagly está me olhando com uma expressão de julgamento profundo...",
        middle: "e com razão, porque esse seu comentário passou de todos os limites da elegância.",
        ending: "Apaga e escreve algo amigável, por favor!"
      },
      {
        intro: "Nossa diplomacia digital está em perigo!",
        middle: "enviar mensagens assim para o mundo virtual só atrai karmas negativos terríveis.",
        ending: "Dê uma chance à paz e reescreva essa mensagem."
      }
    ],
    [
      {
        intro: "Nossa diplomacia acaba de ser severamente testada!",
        middle: "se esse texto fosse um produto, já teria sido recolhido pelo Ministério do Bom Senso.",
        ending: "Mude esse texto agora para algo fofo antes que eu tenha que discursar por 2 horas!"
      },
      {
        intro: "VOCÊ ESTÁ TENTANDO DESTRUIR A REPUTAÇÃO DA NOSSA EQUIPE!",
        middle: "se esse post for publicado, o comitê inteiro vai vir correndo te dar um sermão de 5 horas.",
        ending: "Escolha uma das sugestões pacíficas agora mesmo!"
      },
      {
        intro: "CHEGA DE BRINCADEIRA COM A PAZ SOCIAL!",
        middle: "o Eagly já está abrindo as asas para voar até o seu roteador e cortar os cabos.",
        ending: "Volte atrás e edite esse texto para algo alegre e construtivo!"
      },
      {
        intro: "MEU ELMO NÃO TOLERA TANTA DISCÓRDIA!",
        middle: "você está desafiando diretamente o herói mais patriota e pacífico do universo.",
        ending: "Mude essa frase antes que eu perca a compostura completamente!"
      }
    ],
    [
      {
        intro: "MEU DEUS DO CÉU, VOCÊ NÃO DESISTE NUNCA?!",
        middle: "eu estou aqui tentando salvar a sua reputação social e você insistindo na discórdia...",
        ending: "Quer saber? Quer passar vergonha? Clica de novo que eu deixo, mas vou chorar no canto!"
      },
      {
        intro: "A HUMANIDADE ESTÁ PERDIDA E VOCÊ É A PROVA VIVA!",
        middle: "eu lutei contra invasores alienígenas gigantes e nunca sofri tanto quanto lendo esse seu post.",
        ending: "Vai, clica em enviar de novo se tiver coragem de destruir a internet!"
      },
      {
        intro: "ESTOU COMPLETAMENTE SEM PALAVRAS E CHATEADO!",
        middle: "você escolheu o caminho do caos absoluto e ignorou todas as minhas súplicas de paz.",
        ending: "Clica mais uma vez e liberte o monstro... Mas o Eagly nunca mais vai te dar atenção!"
      },
      {
        intro: "VOCÊ VENCEU, SEU MONSTRO DA INTERNET!",
        middle: "minha alma está em pedaços e o Eagly está escondido debaixo da mesa chorando.",
        ending: "Clica de vez nesse botão maldito e destrua o mundo virtual, eu desisto!"
      }
    ]
  ]
};

function getPersistentMessage(context, level) {
  const contextPool = peacemakerPersistentDialogues[context] || peacemakerPersistentDialogues[PeacemakerContext.SOCIAL];
  const validLevel = Math.min(level, contextPool.length - 1);
  const levelVariations = contextPool[validLevel];
  
  const dialogue = levelVariations[Math.floor(Math.random() * levelVariations.length)];
  return `${dialogue.intro} ${dialogue.middle} ${dialogue.ending}`;
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
    "filmes de comédia leve para rir hoje",
    "como montar um jardim de suculentas em casa",
    "benefícios de tomar chá verde pela manhã",
    "dicas para começar a ler mais livros este ano",
    "melhores trilhas sonoras para relaxar e estudar",
    "como fazer pão caseiro fofinho",
    "lugares bonitos para viajar no fim de semana",
    "como cuidar de plantas de sombra na sala",
    "receita de limonada suíça cremosa",
    "ideias criativas para decoração de quarto",
    "melhores aplicativos de meditação e mindfulness",
    "como fazer origami passo a passo",
    "benefícios de caminhar ao ar livre todos os dias",
    "filmes clássicos de animação para assistir com a família"
  ],
  social: [
    "Acho que todo mundo merece comer um bom hambúrguer hoje! 🍔",
    "Hoje o dia está perfeito para tomar um açaí bem caprichado!",
    "Vocês já viram como o Eagly é uma belezinha? 🦅✨",
    "Mandando muita energia positiva e abraços virtuais para todo mundo!",
    "Vamos focar nas coisas boas da vida e espalhar sorrisos!",
    "Alguém recomenda uma série legal para maratonar no fim de semana?",
    "Paz, amor e muita música boa para todos!",
    "Desejando uma semana maravilhosa e cheia de conquistas para todos! 🌟",
    "Nada como um bom café quentinho para começar o dia bem. ☕",
    "Lembrete amigável: você está indo muito bem, continue assim! 💪❤️",
    "Alguém mais ama dias chuvosos enrolado numa coberta? 🌧️",
    "Espalhe gentileza por onde for, o mundo precisa disso! ✨",
    "Compartilhando boas vibrações para o dia de hoje!",
    "Que tal tirar uns minutinhos hoje para ouvir sua música favorita?",
    "Aproveitem cada momento especial com quem vocês amam! 🥰",
    "Um sorriso pode mudar o dia de alguém. Vamos sorrir mais! 😁",
    "Sextou com muita paz no coração e boas energias!",
    "Quem mais ama o cheirinho de bolo saindo do forno? 🍰",
    "Que a sua semana seja leve, doce e cheia de paz!",
    "Gratidão por todas as pequenas coisas boas de hoje! 🙏"
  ]
};