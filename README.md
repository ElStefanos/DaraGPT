# DaraGPT — Projekat

Projekat je razvijen u **C# jeziku**, i koristi **OpenCL** za hardversku akceleraciju treninga.  
Cilj je da omogući lokalno treniranje jednostavnih jezičkih modela na srpskom jeziku (latinica/ćirilica) bez eksternih biblioteka poput PyTorch-a ili TensorFlow-a.

---

## Kako funkcioniše

- Podaci za trening nalaze se u direktorijumu **`data/`**.  
- Podaci se mogu menjati. Mogu biti bilo kog izvora dok god je `tekstualnog tipa`.
- Svi fajlovi moraju biti **čisti `.txt` fajlovi** – bez metapodataka, oznaka (`<i>`, `<b>` itd.) i nepotrebnih simbola.  
- Za pripremu podataka koristi **Python skripte** za:
  - **čišćenje** (`cleaning`),  
  - **ekstrakciju** (`extraction`),  
  - i **formatiranje** (`formatting`) tekstova.

---

## Tokenizacija

Ako je dataset veći od **2 MB**, preporučuje se korišćenje **`TokenizerCPP`** alata.  
To je brža verzija ugrađenog tokenizera, napisana u **C++**, i radi **3–5x brže** od C# implementacije.  
Ipak, na ogromnim količinama rečenica može i dalje biti spora.

---

## Parametri konfiguracije (`config.json`)

```json
{
  "DModel": 128,
  "NumLayers": 4,
  "ContextSize": 1024,
  "VocabSize": 30200,
  "LearningRate": 0.0015,
  "CheckpointDir": "checkpoints",
  "DevicePreference": "INTEL"
}
```

---

### `DModel`
- Predstavlja **dimenziju modela**, tj. broj neuronskih veza u svakom sloju.  
- Veći broj znači složeniji model koji bolje razume obrasce u jeziku, ali troši više memorije i vremena.  
- Sa `DModel = 128`, model je kompaktan i idealan za eksperimentisanje ili kraće treninge.  
- Za ozbiljniji trening koristi `256–512`, uz uslov da GPU ima dovoljno VRAM-a.

---

### `NumLayers`
- Broj **transformer slojeva** u modelu.  
- Svaki sloj se sastoji od **self-attention** i **feed-forward** delova.  
- Sa `NumLayers = 4`, model ima četiri sloja dubine — dovoljno za razumevanje osnovne strukture rečenica.  
- Veći broj slojeva (8, 12 ili više) povećava sposobnost razumevanja konteksta, ali i vreme treninga.

---

### `ContextSize`
- Broj **tokena** (delova reči) koje model istovremeno posmatra u kontekstu.  
- Veći `ContextSize` = bolji kontekst, ali veće zauzeće RAM/VRAM memorije.  
- `ContextSize = 1024` znači da model vidi do 1024 tokena odjednom, što je vrlo dobra vrednost za tekstove dužine nekoliko rečenica ili paragrafa.  
- Ako koristiš GPU sa manje memorije, smanji vrednost na `512` ili `256`.

---

### `VocabSize`
- Ukupan broj **tokena u vokabularu**.  
- Ova vrednost dolazi direktno iz tokenizera (`Tokenizer` ili `TokenizerCPP`) i tokom treniranja se postavlja automatski nema potrebe menjati ovu vrednost.  
- Sa `VocabSize = 30200`, model može da prepozna oko 30.000 različitih kombinacija reči, delova reči i simbola.  
- Preveliki vokabular usporava model, ali previše mali ograničava razumevanje jezika.  
- Idealno je zadržati između `20.000–60.000` tokena za srpski jezik.

---

### `LearningRate`
- Definiše **brzinu učenja** modela — koliko se težine menjaju nakon svake iteracije.  
- Sa `LearningRate = 0.0015`, model uči umerenim tempom: dovoljno brzo da se vidi napredak, ali ne prebrzo da bi došlo do nestabilnosti.  
- Ako `loss`:
  - **skače** → smanji `LearningRate`  
  - **sporo opada** → blago ga povećaj  
- Optimalan `loss` po završetku epohe obično je između **0.5 i 0.7** Ali na velikim modelima moze biti i viši.
---

### `CheckpointDir`
- Direktorijum gde se čuvaju **fajlovi modela i tokena** tokom treninga.  
- Primer: `checkpoints/model.bin`  
- Checkpoint sadrži težine modela i stanje tokenizera — omogućava da nastaviš trening od tačke gde je prethodni prekinut. **NOTE**: Nastavak treninga još nije implementiran.

---

### `DevicePreference`
- Definiše koji uređaj će se koristiti za **hardversku akceleraciju** (OpenCL backend).  
- Moguće vrednosti:
  - `"INTEL"`
  - `"AMD"`
  - `"NVIDIA"`
  - CPU (fallback ako GPU nije dostupan)  
- Na GPU-u, čak i integrisanom, trening može biti i do **150x brži** nego na CPU-u.  
- Preporuka: uvek koristi GPU koji ima najviše dostupne memorije.

---

## Saveti za optimalan rad

- Za testiranje koristi **manji dataset** (npr. 1–2 MB).  
- Tokom dužeg treninga, prati `loss` i `gradient sum` — stabilnost znači dobar learning rate.  
- Ako koristiš C# tokenizator, za veće korpuse pokreni `TokenizerCPP` pre treninga.  
- Ako koristiš samo CPU, smanji `ContextSize` i `NumLayers` da bi izbegao preopterećenje memorije.

---
