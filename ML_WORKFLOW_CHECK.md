# Check metodologico del workflow Machine Learning

Questo documento descrive le misurazioni, le scelte progettuali e i controlli
del workflow ML per lo sleep staging. È pensato come riferimento per relazione,
presentazione e verifica della riproducibilità.

## 1. Obiettivo sperimentale

Il task è una classificazione multiclasse di epoche PSG da 30 secondi nelle
cinque classi armonizzate:

| Indice | Classe | Nota |
| ---: | --- | --- |
| 0 | Wake | veglia |
| 1 | N1 | sonno leggero e transitorio |
| 2 | N2 | sonno non-REM stabile |
| 3 | N3 | include anche il vecchio stadio N4 |
| 4 | REM | rapid eye movement |

Le etichette `Movement time`, sconosciute o non classificabili vengono escluse.

Dataset di sviluppo: Sleep-EDF Expanded, 100 soggetti e 197 registrazioni
notturne. Dataset esterno: HMC, 151 soggetti nel file attualmente processato.

## 2. Preprocessing e armonizzazione

Tutti i segnali vengono armonizzati a 100 Hz. A questa frequenza il limite di
Nyquist è 50 Hz; le frequenze superiori non sono rappresentabili.

| Modalità | Banda operativa | Motivazione |
| --- | --- | --- |
| EEG | 0,3-35 Hz | attività cerebrale utile allo staging, inclusa beta bassa |
| EOG | 0,1-15 Hz | movimenti oculari lenti e REM, con rimozione della deriva |
| EMG | 10-45 Hz | tono muscolare submentale, margine sotto Nyquist |
| ECG | 0,5-40 Hz | predisposizione disponibile, non usata nel set corrente |

Il filtro è un Butterworth zero-phase di ordine 4 applicato al segnale continuo
prima di resampling ed epoching. Questo evita artefatti ai bordi di ogni epoca.

Controlli:

- [x] stessa frequenza finale per Sleep-EDF e HMC;
- [x] stessa durata di 30 secondi per epoca;
- [x] stessa mappatura a cinque classi;
- [x] N4 unito a N3;
- [x] nessun `NaN` o infinito nei file feature correnti.

## 3. Feature ML

Vengono estratte 24 feature per canale. Con quattro canali si ottengono 96
feature per epoca.

### Dominio temporale, 5 feature per canale

- media;
- deviazione standard;
- RMS;
- ampiezza peak-to-peak;
- zero-crossing rate.

### Dominio frequenziale, 14 feature per canale

La PSD è stimata con Welch su finestre fino a 4 secondi.

- potenza assoluta in delta 0,5-4 Hz;
- theta 4-8 Hz;
- alpha 8-12 Hz;
- sigma 12-16 Hz;
- beta 16-30 Hz;
- potenza relativa nelle stesse cinque bande;
- spectral edge frequency al 95%;
- entropia spettrale;
- rapporto delta/theta;
- rapporto slow/fast.

### Feature non lineari, 3 per canale

- Hjorth activity;
- Hjorth mobility;
- Hjorth complexity.

### Tempo-frequenza, 2 per canale

La STFT misura la variazione della potenza dentro l'epoca:

- media della potenza dei frame;
- deviazione standard della potenza dei frame.

Controlli:

- [x] Sleep-EDF e HMC hanno esattamente gli stessi 96 nomi di feature;
- [x] matrice Sleep-EDF: 457.652 x 96;
- [x] matrice HMC: 137.243 x 96;
- [x] feature finite e allineate a etichette e subject ID.

## 4. Split soggetto-wise

Lo split usa seed 42 e viene eseguito sugli identificativi dei soggetti, mai
sulle singole epoche:

| Split | Soggetti | Ruolo |
| --- | ---: | --- |
| Training | 70 | CV, tuning e fit del classificatore |
| Validation | 15 | calibrazione delle probabilità |
| Test interno | 15 | valutazione finale eseguita una volta |

Tutte le notti dello stesso soggetto rimangono nello stesso split. La funzione
`check_no_subject_overlap` interrompe l'esecuzione se trova un soggetto in più
partizioni.

Scelta: lo split per epoca sarebbe più semplice, ma produrrebbe leakage perché
lo stesso individuo potrebbe comparire sia nel training sia nel test.

## 5. Cross-validation

La 5-fold `GroupKFold` viene eseguita esclusivamente sui 70 soggetti di training.
Ogni iterazione usa circa 56 soggetti per il fit e 14 soggetti per la validation
CV. Vengono quindi effettuati cinque fit indipendenti.

Per ciascun fold si misurano:

- Macro-F1 sul training del fold;
- Macro-F1 sui soggetti di validation del fold;
- balanced accuracy;
- Cohen's kappa;
- metriche per classe e matrice di confusione.

La media delle cinque validation CV misura la generalizzazione a soggetti non
usati nel rispettivo fit. Non è il risultato sul test interno.

## 6. Modelli confrontati

| Modello | Ruolo | Scelte principali |
| --- | --- | --- |
| Logistic Regression multinomiale | baseline lineare interpretabile | StandardScaler, `lbfgs`, max_iter 2000 |
| Random Forest | ensemble bagging non lineare | profondità e dimensione foglie regolarizzate |
| XGBoost | boosting regolarizzato | alberi bassi, gamma, min-child e penalità L2 |

Griglie correnti:

- LogReg: `C` in 0,1; 0,5; 1; 2;
- RF: 300/500 alberi, profondità 12/18, almeno 2/5 campioni per foglia;
- XGBoost: 300/500 alberi, profondità 3/4, learning rate 0,05,
  `min_child_weight` 3/7, `gamma` 0,1 e `reg_lambda` 5.

## 7. Gestione dello sbilanciamento

### Class weight

Il peso di una classe è inversamente proporzionale alla sua frequenza:

```text
peso_c = numero totale / (numero classi * campioni della classe c)
```

Non vengono creati nuovi campioni. RF e LogReg usano `class_weight`; per
XGBoost multiclass gli stessi pesi vengono applicati alla loss tramite
`sample_weight`, ricalcolato esclusivamente sul training del fold.

### Half-SMOTE

SMOTE interpola feature di campioni minoritari vicini. Nel progetto ogni classe
minoritaria viene portata al massimo al 50% della classe maggioritaria.

Sul training fisso corrente:

| Quantità | Numero |
| --- | ---: |
| Campioni originali | 324.013 |
| Campioni sintetici | 298.578 |
| Totale dopo half-SMOTE | 622.591 |
| Moltiplicatore | 1,92x |

SMOTE viene applicato dentro ogni training fold e una volta sul training finale.
Applicarlo prima della CV permetterebbe a informazioni della validation di
entrare nella generazione dei sintetici e provocherebbe leakage.

I sintetici esistono solo in RAM: NPZ, etichette e subject ID originali non
vengono modificati.

## 8. Tuning con vincolo di overfitting

Per ogni configurazione:

```text
gap = Macro-F1 medio training - Macro-F1 medio validation CV
```

Regola progettuale:

```text
gap <= 0,10  -> configurazione eleggibile
gap > 0,10   -> configurazione classificata come overfitted
```

Tra le configurazioni eleggibili viene scelta quella con Macro-F1 CV più alto.
Se nessuna rispetta il vincolo, viene usata come fallback quella con gap minore
e il mancato rispetto del vincolo viene registrato. La soglia 0,10 è euristica,
non una legge statistica: test interno, deviazione dei fold e HMC restano
necessari.

## 9. Fit finale e calibrazione

Dopo la selezione, il classificatore viene addestrato su tutti i 70 soggetti di
training. Il modello viene poi congelato e la calibrazione sigmoid/Platt viene
fittata sui 15 soggetti di validation.

La calibrazione tenta di rendere la probabilità dichiarata coerente con la
frequenza empirica degli errori. Le misure principali sono:

- ECE: differenza tra confidenza e accuratezza nei bin; minore è meglio;
- Brier multiclass: errore quadratico delle probabilità; minore è meglio;
- reliability diagram: confidenza prevista contro accuratezza empirica.

La calibrazione non è automaticamente benefica. Per la precedente LogReg ha
peggiorato ECE da circa 0,016 a 0,194 e ha portato F1 N1 e REM a zero dopo
`argmax`. Per questo la configurazione corrente usa LogReg senza calibrazione,
con le probabilità native. Per gli altri modelli raw e calibrated vengono
conservati e confrontati.

## 10. Metriche finali

| Metrica | Cosa misura | Direzione |
| --- | --- | --- |
| Accuracy | quota totale di predizioni corrette | più alta |
| Balanced accuracy | media del recall delle cinque classi | più alta |
| Macro-F1 | media F1 dando uguale peso a ogni classe | più alta |
| Weighted-F1 | media F1 pesata per prevalenza | più alta |
| F1 per classe | equilibrio precision-recall della singola fase | più alta |
| Cohen's kappa | accordo oltre quello atteso per caso | più alta |
| ROC-AUC OvR | capacità di ordinare una classe contro le altre | più alta |
| AUPRC OvR | precision-recall, utile con classi rare | più alta |
| ECE | errore di calibrazione per bin | più bassa |
| Brier | errore quadratico delle probabilità | più bassa |

Macro-F1 e balanced accuracy sono prioritarie rispetto all'accuracy perché N1,
N3 e REM sono meno frequenti. ROC-AUC non va interpretata da sola: una classe
può avere AUC alta ma F1 basso se il ranking è buono e la decisione multiclass
`argmax` è sfavorevole.

## 11. Risultati interni correnti

| Configurazione | CV Macro-F1 | Test Macro-F1 | Gap | Stato |
| --- | ---: | ---: | ---: | --- |
| RF + class weight | 0,738 | 0,738 | 0,185 | overfitted |
| XGBoost + class weight | 0,732 | 0,756 | 0,067 | valido |
| LogReg + class weight, calibrata | 0,718 | 0,478 | 0,025 | valida ma calibrazione dannosa |
| XGBoost + half-SMOTE | 0,755 | 0,751 | 0,072 | valido |

Scelta secondo protocollo train-only: XGBoost + half-SMOTE, perché ha il miglior
Macro-F1 CV tra le configurazioni che rispettano il vincolo. XGBoost + class
weight resta la baseline principale: sul test interno ha balanced accuracy e
Macro-F1 leggermente migliori, mentre half-SMOTE è migliore in CV. La differenza
va quindi verificata con HMC e LOSO, senza cambiare retroattivamente il criterio
di selezione in base al test.

## 12. Validazione esterna HMC

Il modello già addestrato viene applicato alle feature HMC senza retuning e
senza modificare preprocessing o soglie usando HMC. Questo misura il domain
shift tra dataset. HMC non deve essere usato per scegliere iperparametri.

Si riportano le stesse metriche del test interno, con particolare attenzione a
Macro-F1, balanced accuracy, F1 per classe e calo rispetto a Sleep-EDF.

## 13. LOSO

LOSO ripete:

```text
99 soggetti training -> 1 soggetto test
```

per tutti i 100 soggetti. È una misura di robustezza inter-soggetto. Non produce
direttamente lo stesso gap di overfitting della GroupKFold; è più informativo
riportare Macro-F1 aggregato, media/deviazione per soggetto e distribuzione dei
risultati. Class weight è l'opzione sostenibile; half-SMOTE richiederebbe SMOTE
e training XGBoost cento volte.

## 14. Explainability e figure

- confusion matrix: quali classi vengono confuse;
- F1 per classe: evidenzia il limite su N1;
- ROC OvR: separabilità a tutte le soglie;
- PR OvR: più informativa per classi rare;
- reliability diagram: qualità delle probabilità;
- SHAP globale: importanza media assoluta delle feature.

Attualmente SHAP salva il PNG globale. I vettori SHAP numerici non sono ancora
salvati su disco; le NPZ `ml_test_probs_*` contengono invece predizioni e
probabilità raw/calibrate del test.

## 15. Output e tracciabilità

| Percorso | Contenuto |
| --- | --- |
| `results/tables/ml_train_sleep.csv` | confronto scalare delle run train/internal |
| `results/logs/ml_metrics_<run>.json` | metriche dettagliate e protocollo |
| `results/logs/ml_model_<run>.pkl` | estimatore addestrato/calibrato |
| `results/logs/ml_test_probs_<run>.npz` | label, predizioni e probabilità test |
| `results/figures/<run>/` | figure diagnostiche e SHAP |
| `results/logs/loso_metrics_<run>.json` | risultati LOSO aggregati |

## 16. Checklist finale prima della relazione

- [x] split e fold soggetto-wise;
- [x] controllo esplicito contro subject leakage;
- [x] preprocessing e feature identici tra Sleep-EDF e HMC;
- [x] tuning eseguito solo sui soggetti training;
- [x] class weighting ricalcolato nel training fold;
- [x] SMOTE applicato solo nel training fold;
- [x] vincolo di overfitting registrato;
- [x] raw e calibrated salvati separatamente;
- [x] test interno non usato per tuning;
- [ ] completare e riportare validazione esterna HMC per il modello finale;
- [ ] completare LOSO e riportare distribuzione per soggetto;
- [ ] salvare i vettori SHAP se servono analisi successive senza ricalcolo.
