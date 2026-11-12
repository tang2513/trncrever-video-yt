# 🎙️ Transcritor de Vídeos Longos

Aplicativo local (Python + Streamlit + faster-whisper) para transcrever **vídeos e gravações de horas**
— de arquivos do seu computador, links do **YouTube** e **VODs da Twitch** — em texto fiel, com
marcações de tempo, em **TXT, SRT e VTT**.

* **100 % local:** nada é enviado para serviços de IA nem para servidores de terceiros. A internet só é usada
  para baixar o modelo (uma vez) e, se você escolher YouTube/Twitch, para baixar legendas/áudio.
* **Feito para vídeos de 2 a 4 horas:** o áudio é convertido, dividido em blocos de ~10 minutos (cortando em
  pausas de silêncio) e transcrito um bloco por vez, direto do disco — a memória RAM não cresce com a duração.
* **Não perde o progresso:** cada bloco concluído fica salvo; se o programa fechar ou o computador reiniciar,
  a tarefa é retomada de onde parou.
* **Interface sempre responsiva:** a transcrição roda em um processo separado (o *worker*); você pode fechar
  ou recarregar a página sem interromper o trabalho.

> ⚠️ **Estado de validação — leia antes de confiar:** o código e os testes foram escritos em um computador
> **sem Python instalado**, portanto **nada foi executado** (nem o aplicativo, nem os testes). Foram feitas apenas
> verificações estáticas (nomes importados × definidos, assinaturas de chamadas, balanceamento de parênteses) e o
> formato da saída do FFmpeg foi conferido com comandos reais. **Nenhum vídeo de 2–4 horas foi testado.**
> Rode `python -m pytest` (seção [Testes](#-testes)) e reporte qualquer falha; veja
> [o que ainda precisa ser validado](#-o-que-ainda-precisa-ser-validado-no-seu-computador).

---

## Índice

1. [Requisitos](#-requisitos)
2. [Instalação passo a passo (Windows)](#-instalação-passo-a-passo-windows)
3. [Como usar](#-como-usar)
4. [Resultados gerados](#-resultados-gerados)
5. [Como funciona por dentro](#-como-funciona-por-dentro)
6. [Opções avançadas](#-opções-avançadas)
7. [Solução de problemas](#-solução-de-problemas)
8. [Testes](#-testes)
9. [O que ainda precisa ser validado](#-o-que-ainda-precisa-ser-validado-no-seu-computador)

---

## 📋 Requisitos

| Item | Detalhe |
|---|---|
| Sistema | Windows 10 ou 11 (64 bits). O núcleo roda em outros sistemas, mas o seletor de arquivos nativo é só do Windows. |
| Python | **3.11 a 3.14 (64 bits)** — testado de fato apenas com 3.14.10 |
| FFmpeg | Instalado separadamente e acessível pelo `PATH` (o aplicativo verifica sozinho) |
| RAM | 8 GB recomendados. Com a CPU: `base` ≈ 1 GB, `small` ≈ 1,5 GB, `medium` ≈ 3 GB, `large-v3` ≈ 5–6 GB livres |
| Disco | Cerca de **2× a duração do áudio em WAV** + o tamanho do áudio baixado. Para 4 h: ~ 0,5 GB (WAV 16 kHz mono) → reserve alguns GB; os temporários são apagados ao concluir |
| GPU (opcional) | Somente **NVIDIA** com CUDA 12. **Placas AMD e Intel não são suportadas** pelo motor de transcrição — nesses casos o aplicativo usa a CPU automaticamente |

---

## 🛠️ Instalação passo a passo (Windows)

### 1. Instalar o Python 3.11

1. Acesse <https://www.python.org/downloads/windows/> e baixe o **Windows installer (64-bit)** da versão **3.11.x**.
2. Execute o instalador. **Marque a caixa “Add python.exe to PATH”** (na primeira tela) e clique em **Install Now**.
3. Abra um **novo** terminal (PowerShell) e confira:

   ```powershell
   python --version
   ```

   Deve mostrar `Python 3.11.x`.

> **Apareceu “Python was not found; run without arguments to install from the Microsoft Store”?**
> É o atalho da Loja, que atrapalha. Vá em **Configurações → Aplicativos → Configurações avançadas de aplicativo →
> Aliases de execução de aplicativo** e **desligue** `python.exe` e `python3.exe`. Feche e abra o terminal.
> Se você tem várias versões, também funciona `py -3.11` no lugar de `python`.

### 2. Instalar o FFmpeg

**Opção A — pelo winget (mais simples):**

```powershell
winget install Gyan.FFmpeg
```

Feche e abra o terminal e vá ao passo de verificação abaixo.

**Opção B — manualmente:**

1. Baixe <https://www.gyan.dev/ffmpeg/builds/> → **ffmpeg-release-essentials.zip**.
2. Extraia o conteúdo em `C:\ffmpeg` de modo que exista o arquivo **`C:\ffmpeg\bin\ffmpeg.exe`**
   (se o zip criar uma subpasta com a versão, mova o conteúdo dela para `C:\ffmpeg`).
3. Adicione `C:\ffmpeg\bin` ao **PATH**:
   1. Tecle <kbd>Windows</kbd>, digite **“Editar as variáveis de ambiente do sistema”** e abra.
   2. Clique em **Variáveis de Ambiente…**
   3. Em **Variáveis de usuário**, selecione **Path** → **Editar…** → **Novo** → digite `C:\ffmpeg\bin` → **OK** em todas as janelas.
4. **Feche e abra um novo terminal** (o PATH só vale para terminais novos).

**Verificação (vale para as duas opções):**

```powershell
ffmpeg -version
ffprobe -version
```

Os dois comandos precisam mostrar a versão. Se `ffmpeg` não for reconhecido, o PATH não foi aplicado — confira o
caminho e abra um terminal novo. (O aplicativo também mostra o status do FFmpeg na barra lateral.)

### 3. Preparar a pasta e o ambiente virtual

Copie a pasta do projeto (`transcritor`) para onde quiser e abra o PowerShell nela:

```powershell
cd D:\claude\transcritor
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o prompt começa com `(venv)`.

> **Erro “a execução de scripts foi desabilitada neste sistema”?** Rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` e tente de novo.
> Ou use o Prompt de Comando (cmd) com `venv\Scripts\activate.bat`.

Sempre que abrir um terminal novo para usar o aplicativo, ative o ambiente de novo (`.\venv\Scripts\Activate.ps1`).

### 4. Instalar as dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Isso instala Streamlit, faster-whisper, CTranslate2, yt-dlp, psutil e srt (pode levar alguns minutos).

**Somente se você tem uma GPU NVIDIA** (driver atualizado) e quer acelerar:

```powershell
python -m pip install -r requirements-gpu.txt
```

### 5. Iniciar o aplicativo

```powershell
streamlit run app.py
```

O navegador abre em <http://localhost:8501>. (Se não abrir, copie o endereço para o navegador.)
Para encerrar o aplicativo, feche o terminal ou tecle <kbd>Ctrl</kbd>+<kbd>C</kbd> nele — **uma transcrição em
andamento continua rodando em segundo plano** e pode ser reencontrada ao abrir o aplicativo de novo.

> Na **primeira** transcrição com cada modelo, ele é baixado da internet (`base` ≈ 150 MB, `small` ≈ 500 MB,
> `medium` ≈ 1,5 GB, `large-v3` ≈ 3 GB) para a pasta `models/`. Nas próximas vezes não é preciso internet.

---

## 🧭 Como usar

### Transcrever

1. **Origem**
   * **📁 Arquivo local:** cole o caminho completo (`C:\Videos\gravacao.mp4`) ou clique em **📂 Selecionar arquivo…**
     (abre o explorador do Windows). Formatos: MP4, MKV, MOV, AVI, WEBM, MP3, WAV, M4A, FLAC.
     O arquivo é lido direto do disco e **nunca é modificado nem apagado**.
   * **▶️ YouTube:** cole o link do vídeo. Escolha o que fazer com as legendas:
     * *Priorizar legendas em português* (padrão): se houver legenda em português (do criador ou automática no
       idioma original), o texto é extraído **sem baixar o áudio nem transcrever**; se não houver, o app avisa
       “Este vídeo não possui legendas em português” e transcreve o áudio.
     * *Usar legendas em outros idiomas*: aceita legenda de outro idioma se não houver em português.
     * *Ignorar legendas*: sempre baixa o áudio e transcreve.

     > Legendas **traduzidas automaticamente** pelo YouTube não são usadas (não são fiéis à fala); só a legenda
     > automática no idioma original do vídeo.
   * **🟣 Twitch (VOD):** cole o link da gravação (`https://www.twitch.tv/videos/123456789`). Transmissões ao vivo
     e clipes não são suportados.
   * **Conteúdo que exige login:** em *“Conteúdo que exige login”* informe o caminho de um `cookies.txt`
     (formato Netscape) exportado **da sua própria sessão autorizada**. O arquivo é copiado para a pasta da tarefa e
     apagado ao final; o original não é alterado. O app **não** contorna DRM, assinaturas pagas nem restrições de acesso.
2. **Idioma:** *Português* (padrão), detecção automática ou outro idioma. Com detecção automática o idioma é
   identificado no primeiro bloco com fala e mantido nos demais.
3. **Modelo** (o padrão é `small`) — modelos maiores costumam ser mais precisos, mas exigem mais memória e tempo:

   | Modelo | Velocidade | Quando usar |
   |---|---|---|
   | `base` | muito rápido | testes, áudio limpo, computadores modestos |
   | `small` | rápido | **equilíbrio recomendado** para a maioria dos casos |
   | `medium` | lento na CPU | mais precisão; ideal com GPU NVIDIA |
   | `large-v3` | muito lento na CPU | máxima precisão; exige muita RAM/VRAM |

4. **Dispositivo:** *Automático* (GPU NVIDIA quando utilizável, senão CPU), *CPU* ou *GPU/CUDA*. Se escolher GPU e ela
   não estiver disponível, o app explica o motivo e oferece **Processar pela CPU**.
5. **Marcações de tempo no TXT:** a cada 5, 10 ou 15 minutos, ou a cada segmento de fala.
6. Clique em **🚀 Iniciar transcrição**.

### Acompanhar, cancelar e retomar

* A tela de andamento mostra a etapa atual, o progresso **real**, blocos concluídos/total, tempo decorrido, tempo
  restante (estimado a partir da velocidade medida — só aparece depois de alguns blocos), modelo, dispositivo e
  mensagens. As 9 etapas: verificação → legendas → download → conversão → pausas/blocos → modelo → transcrição →
  organização → finalização.
* **⛔ Cancelar processamento:** encerra com segurança (inclusive FFmpeg/yt-dlp) e **preserva os blocos já concluídos**.
* **Retomar:** na aba **Tarefas** (ou no aviso da tela inicial) clique em **▶ Retomar**. Só os blocos que faltam
  são processados.
* Só **uma** transcrição roda por vez (duas disputariam CPU, memória e GPU).

### Depois de concluir

Pré-visualização do começo do texto, botões **⬇️ Baixar .txt/.srt/.vtt**, **📋 Copiar texto completo**,
**📂 Abrir pasta** e **➕ Nova transcrição**.

---

## 📦 Resultados gerados

```
transcricoes/
└── nome_do_video/
    ├── nome_do_video.txt    texto organizado em parágrafos, com cabeçalho e marcações de tempo
    ├── nome_do_video.srt    legendas
    └── nome_do_video.vtt    WebVTT
```

* Tudo em UTF-8. Os tempos são sempre do **vídeo original**.
* **Nada é sobrescrito:** se a pasta já existir, é criada `nome_do_video (2)`. (Há uma opção avançada, com confirmação
  dupla, para substituir.)
* Cabeçalho do TXT: título, origem, fonte, método (transcrição ou legenda do YouTube), data de processamento, idioma,
  modelo, dispositivo e duração.
* O texto **não é resumido, corrigido nem enxugado**. Frases repetidas pelo palestrante são mantidas; trechos
  silenciosos ficam sem texto (nada é inventado para preenchê-los).

Arquivos de trabalho (em `temp/tasks/<ID da tarefa>/`) são apagados ao concluir, exceto o progresso e os trechos
transcritos. Logs ficam em `logs/` (`app.log` e `worker_<ID>.log`).

---

## ⚙️ Como funciona por dentro

```
 app.py (Streamlit)                       src/worker.py (processo independente)
 ───────────────────                      ──────────────────────────────────────
 cria config.json + checkpoint     ─────► subprocess.Popen([python, -m, src.worker, --task-id, ID])
 lê checkpoint.json a cada 1,5 s   ◄───── grava checkpoint.json / manifest.json (atômico, com lock)
 escreve cancel.json ao cancelar   ─────► consulta cancel.json em pontos seguros
```

1. **Validação** (arquivo, link, FFmpeg, espaço em disco).
2. **YouTube:** consulta as legendas (`yt-dlp`); se houver legenda adequada, o texto é extraído e o fluxo pula para a organização.
3. **Download** somente do áudio (`yt-dlp`, com novas tentativas e intervalos progressivos para falhas de rede).
4. **Conversão** com FFmpeg para WAV 16 kHz, mono (em fluxo, sem carregar o arquivo na RAM).
5. **Divisão em blocos** de ~10 min: o `silencedetect` (`noise=-35dB:d=0.4`, ajustável) é executado só em janelas
   ao redor de cada limite nominal; o corte vai para o meio da melhor pausa (ou para o limite nominal se não houver).
   Os blocos são definidos por **amostras inteiras**: o fim de um bloco é exatamente o início do próximo — sem lacunas,
   sem sobreposição, sem precisar comparar texto. Os pontos reais de corte ficam no `manifest.json`.
6. **Transcrição** com `faster-whisper` (`vad_filter=True`, `condition_on_previous_text=False`, fallback de
   temperatura e limiares contra repetição/alucinação), um bloco por vez; cada bloco é salvo em
   `segments/segments_NNNN.json` e só então marcado como concluído no manifesto e no checkpoint.
7. **Exportação** TXT/SRT/VTT lendo os blocos do disco um a um.

**Quem escreve o quê:** a interface escreve `config.json`, `cancel.json` e o checkpoint inicial (e reinicia o estado
ao retomar); o worker escreve o restante do checkpoint e todo o manifesto. Toda escrita no checkpoint usa lock de arquivo
e gravação atômica (`os.replace`), então o último estado válido nunca se perde e os dois lados nunca escrevem juntos.

**Processos:** o worker é identificado por PID + horário de criação + ID da tarefa na linha de comando (o PID sozinho
pode ser reaproveitado pelo Windows). Se o worker morrer, a interface marca a tarefa como interrompida e encerra
FFmpeg/yt-dlp órfãos.

```
transcritor/
├── app.py                     interface Streamlit
├── requirements.txt           dependências (fixadas)  ·  requirements-gpu.txt (NVIDIA)  ·  requirements-dev.txt (testes)
├── pytest.ini
├── .streamlit/config.toml
├── src/
│   ├── hardware.py            CPU/RAM/GPU, verificação real do CUDA, escolha do dispositivo e fallback
│   ├── utils.py               erros amigáveis, JSON atômico, locks, processos, validações, timestamps
│   ├── subtitles.py           VTT/SRT, escolha de legenda, parágrafos, TXT/SRT/VTT
│   ├── downloader.py          validação de links, yt-dlp (legendas e áudio)
│   ├── audio_processor.py     FFmpeg, extração, silencedetect, plano e gravação dos blocos
│   ├── transcription.py       faster-whisper por bloco
│   ├── task_manager.py        tarefas, checkpoint, cancelamento, controle do worker
│   ├── file_picker.py         seletor de arquivos do Windows (PowerShell em processo isolado)
│   └── worker.py              pipeline em processo independente
├── tests/                     testes automatizados
├── transcricoes/              resultados
├── temp/tasks/                tarefas (config, checkpoint, manifesto, temporários)
├── logs/                      logs
└── models/                    modelos baixados
```

---

## 🔧 Opções avançadas

Em **⚙️ Opções avançadas** na tela inicial:

* **Tamanho dos blocos** (5–20 min), **janela de busca por pausas**, **limite de silêncio (dB)** e **duração mínima da
  pausa** — se o áudio tiver ruído de fundo forte e os cortes não caírem em pausas, suba o limite (ex.: −30 dB);
  se cortar no meio de falas baixas, desça (ex.: −42 dB). O corte nunca descarta áudio: só escolhe *onde* dividir.
* **Beam size** (1–10): maior = um pouco mais preciso e mais lento.
* **Manter temporários**, **substituir resultado anterior** (com confirmação).

Variável de ambiente `TRANSCRITOR_DATA_DIR`: move `transcricoes/`, `temp/`, `logs/` e `models/` para outra pasta
(útil para guardar tudo em outro disco). Defina antes de iniciar:

```powershell
$env:TRANSCRITOR_DATA_DIR = "E:\dados_transcritor"
streamlit run app.py
```

---

## 🩺 Solução de problemas

| Sintoma | O que fazer |
|---|---|
| `python` não é reconhecido / abre a Microsoft Store | Reinstale marcando **Add python.exe to PATH** e/ou desligue os *aliases de execução* (veja o passo 1). Abra um terminal novo. |
| `Activate.ps1` bloqueado | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` ou use o `cmd` com `activate.bat`. |
| `pip install` falha ao compilar algo | Atualize o pip (`python -m pip install --upgrade pip`) e confirme que é **Python 3.11 de 64 bits** (`python -c "import struct;print(struct.calcsize('P')*8)"` deve mostrar 64). |
| Barra lateral: “FFmpeg não encontrado” | Confira `ffmpeg -version` em um terminal **novo**; revise o PATH (`C:\ffmpeg\bin`). Clique em **🔄 Verificar hardware novamente**. |
| `streamlit` não é reconhecido | Ative o ambiente virtual ou use `python -m streamlit run app.py`. |
| Porta 8501 ocupada | `streamlit run app.py --server.port 8502` |
| Botão “Selecionar arquivo…” não abre nada | Pode estar atrás do navegador (olhe a barra de tarefas) ou bloqueado por política do Windows. Digite/cole o caminho no campo — funciona igual. |
| Erro ao baixar o modelo | Verifique a internet/proxy/antivírus. O app tenta novamente 3 vezes. Se a rede for restrita, baixe o modelo em outro computador e copie para `models/`. |
| “GPU/CUDA não disponível” | Placas **AMD/Intel não funcionam** com este motor — use CPU (é o comportamento automático). Em NVIDIA: atualize o driver e instale `requirements-gpu.txt`; mensagens sobre `cublas64_12.dll`/`cudnn` indicam que esse passo falta. |
| Falha de GPU no meio do trabalho | Na tela da tarefa clique em **▶ Retomar pela CPU** — os blocos prontos são mantidos. |
| “Memória insuficiente” | Feche programas pesados ou escolha um modelo menor (`small`/`base`). |
| Muito lento | Na CPU, `medium` e `large-v3` podem levar muitas horas para vídeos longos. Use `small`/`base` e meça com um trecho curto: o app mostra o tempo restante estimado depois dos primeiros blocos. |
| “Não há espaço suficiente no disco” | Libere espaço no disco do projeto (ou use `TRANSCRITOR_DATA_DIR` para outro disco). |
| YouTube: erro de download / “Sign in to confirm you’re not a bot” | `python -m pip install -U yt-dlp` (o YouTube muda com frequência). Para conteúdo que exige login, informe um `cookies.txt` da sua sessão. |
| Twitch: “indisponível / expirado” | VODs expiram e podem ser privados ou só para assinantes. Não é possível baixar conteúdo protegido. |
| “O processamento foi encerrado antes de terminar” | Programa fechado/computador reiniciado: abra **Tarefas** e clique em **▶ Retomar**. |
| Tarefa “travada” | Veja o final do log em *Detalhes técnicos* na tela da tarefa ou em `logs/worker_<ID>.log`. |
| Texto com repetições/invenções em trechos de ruído | Use um modelo maior, ajuste o limite de silêncio e confira se o áudio original não tem trechos corrompidos. |
| Quero apagar tarefas antigas | Aba **Tarefas → Excluir** (remove só os temporários; resultados e arquivos originais não são tocados). |

---

## 🧪 Testes

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

| Arquivo | Cobre |
|---|---|
| `test_utils.py` | validação de caminhos, formatos, timestamps, JSON atômico (corrupção, falha na troca, escrita concorrente), locks, identificação de processos (PID reaproveitado), execução/cancelamento de processos, novas tentativas, mensagens de erro |
| `test_hardware.py` | identificação de GPU/CUDA (CTranslate2 simulado), escolha de dispositivo, **fallback para CPU**, avisos de RAM |
| `test_downloader.py` | validação de links YouTube/Twitch, classificação dos erros do yt-dlp, progresso, cookies |
| `test_subtitles.py` | leitura de VTT (inclusive o formato “rolante” do YouTube) e SRT, escolha de legenda, parágrafos e marcações, geração de TXT/SRT/VTT |
| `test_audio_processor.py` | FFmpeg, extração de áudio, `silencedetect`, planejamento e gravação dos blocos (**sem lacunas/duplicações**, propriedade testada com durações aleatórias) |
| `test_transcription.py` | parâmetros do faster-whisper, conversão de timestamps, carregamento, novas tentativas de rede, fallback GPU→CPU (modelo simulado) |
| `test_task_manager.py` | criação, checkpoint (concorrência entre threads **e processos**), progresso, cancelamento, worker morto, órfãos, retomada, **inicialização real do worker** |
| `test_worker.py` | pipeline completo com FFmpeg real e motor simulado: resultados, retomada, cancelamento, erros, YouTube (legendas/áudio) |
| `test_file_picker.py` | seletor de arquivos (processo simulado) |
| `test_integration.py` | **opcional**: modelo real (`RUN_INTEGRATION=1`) |

Testes que precisam do FFmpeg são pulados se ele não estiver instalado. Opcionais:

```powershell
$env:RUN_INTEGRATION = "1"     # usa o faster-whisper de verdade (baixa o modelo base)
$env:TRANSCRITOR_TEST_AUDIO = "C:\caminho\fala_curta.wav"   # opcional: valida o texto de uma fala real
$env:RUN_SLOW = "1"            # processa 1 hora de áudio sintético e mede o uso de RAM
python -m pytest -s
```

---

## ✅ O que ainda precisa ser validado no seu computador

1. **Rodar `python -m pytest`** e corrigir/relatar falhas — a suíte nunca foi executada pelo autor.
2. **Uma transcrição real curta** (`RUN_INTEGRATION=1`, ou pelo app) para confirmar download do modelo, VAD e qualidade.
3. **Um vídeo longo de verdade (2–4 h)**: tempo total, uso de RAM (o app mostra o pico do worker) e retomada após
   fechar o programa no meio. `RUN_SLOW=1` valida apenas a parte de áudio (blocos/memória) com 1 hora sintética.
4. **Ambiente virtual no Windows:** o `python.exe` do venv é um lançador que cria um processo filho; a detecção do worker
   foi escrita para isso, mas precisa ser confirmada na prática (cancelar/retomar/fechar e reabrir a página).
5. **GPU NVIDIA** (se houver): este desenvolvimento foi feito em uma máquina com GPU AMD, então o caminho CUDA só foi
   testado com simulações.
6. **YouTube/Twitch reais:** os testes usam respostas simuladas do yt-dlp (sem rede).
