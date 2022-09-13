# Image processing
Repository del tema d'anno di image processing

# Guida all'uso

### Prerequisiti

Per poter utilizzare il sistema è necessario disporre di una macchina con Python
3.8 installato nel PATH di sistema, insieme a pip per installare le dipendenze.

### Installazione dipendenze
Le dipendenze sono listate in `requirements.txt` (`requirements-win.txt` per i
sistemi operativi windows).
Si può avviare l'installazione con:

```
$: pip install -r requirements.txt
```
(sostituire requirements.txt con la versione windows se necessario)

### Avvio del progetto
Una volta installate le librerie, il programma può essere avviato eseguendo
`main.py`. Per ottenere la lista delle opzioni con cui può essere avviato è
sufficente avviare lo script con il flag `-h`:

```
$: python3 main.py -h
```

### Walkthrough demo
Per poter avviare una demo è necessario dapprima scegliere una sorgente video
con inquadratura fissa. Si può specificare una (e solo una) delle seguenti sorgenti:

1. --directory DIRECTORY: utilizza come sorgente video le immaigini contenute nella directory specificata (path specificato in DIRECTORY)
2. --camera CAMERA: utilizza come sorgente video una delle videocamere collegate al computer(CAMERA è un indice numerico che le identifica in cv2);
3. --image IMAGE: utilizza una singola immagine come sorgente video (IMAGE è il suo path);
4. --video VIDEO: utilizza un video come sorgente (VIDEO è il suo path);

Nel repository è presente una directory di test a scopo dimostrativo. All'interno
ci sono 7 immagini, una per la taratura e 6 con soggetti posti a uno o due metri di distanza.

Inoltre è necessario fornire al sistema una rete neurale addestrata basata su YOLO in
formato compatibile con pytorch. Si possono ottenere delle versioni pre-addestrate
a sul repository ufficiale (https://github.com/ultralytics/yolov5)[https://github.com/ultralytics/yolov5]. Il repository ne include una versione di piccole dimensioni a scopo dimostrativo.

Per avviare il setup (taratura) bisogna richiamare l'azione specificando le dimensioni
della scacchiera presente nell'immagine di calibrazione:

```
$: python3 main.py setup --checkerboard-cols 9 --checkerboard-rows 6 --checkerboard-size 25 --directory "test/"
```

Verrà visualizzata una finestra che mostra la posizione della scacchiera rilevata.
Il setup generà i dati per la correzione prospettica e li salverà in `homography.bin`.
Una volta generati questi dati, si può eseguire il task di rilevamento distanza:

```
$: python3 main.py detect --directory 'test/' --model 'checkpoints/pretrained.pt'
```

In output verrà mostrata una finestra con quattro immagini: l'originale, l'immagine
con la prospettiva corretta, una immagine che indica le persone rilevate, una immagine
con le distanze rilevate dall software.

Per chiudere la finestra, premere `q`.

