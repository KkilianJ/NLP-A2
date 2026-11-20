# Project 2: IMDb sentiment classification with GloVe

I started by checking whether GloVe could represent the words in IMDb reviews. Then I tried several ways to handle words that GloVe missed, compared pooling with recurrent encoders, and looked at which reviews the models got wrong. This README follows that order. The results are for my GloVe track.

**The main result:** Fine-tuned MeanPool was my best GloVe system. It reached **88.77% validation Macro F1** across three seeds and **89.11% held-out test Macro F1** with seed 42. BiGRU made a substantial difference when GloVe was frozen. Once both systems could update the word vectors, MeanPool was slightly ahead.

## 1. Data and evaluation

The task is to classify reviews as positive or negative. For the initial coverage analysis, I used only the **24,904 reviews in the published `mteb/imdb` training split**. For the classification experiments, I reused **Project 1's own split**, which pooled the published partitions and then made a stratified 70/15/15 split with seed 42.

| Partition | Reviews | How I used it |
|---|---:|---|
| Project 1 training | 34,707 | The early MLP baselines trained on this whole partition. Later experiments split it into 31,236 fitting reviews and 3,471 inner-validation reviews. |
| Project 1 validation | 7,437 | Compared systems on the same reviews. |
| Project 1 held-out test | 7,438 | Evaluated fixed seed-42 models after checkpoint selection. |

The dataset revision was `d05f0155defa7991dad75bc68c5ccb6774b1fdc5`. I verified its row order and labels against Project 1's fingerprint. The later inner split leaves 10% of the Project 1 training partition for selecting settings and checkpoints, with stratification and seed 42. This is a fixed development policy. It also means comparisons with Project 1's historical scores are system-level comparisons, with different model-fitting procedures.

I used **Macro F1** as the main metric because it gives positive and negative reviews equal weight and matches Project 1. I also report accuracy for the final comparisons. The PyTorch classifiers use a single logit with binary cross-entropy and a fixed **0.5 probability threshold**. Seeds **42, 43 and 44** measure training variation while keeping the data split fixed. The test comparison uses **seed 42**, following the original run, without selecting the best seed. The held-out split here is **Project 1's test split**.

## 2. First I checked how to tokenize for GloVe

I chose **GloVe 6B** because it provides uncased English word vectors and I could check word-level compatibility directly on IMDb. I started at **100 dimensions** as a manageable common setting across experiments. The dimension experiment in Section 6 later tested that choice.

I decoded HTML entities, removed tags, lowercased the text and normalised curly apostrophes. I excluded punctuation-only tokens from coverage counts and kept low-frequency words. A token counted as covered if its complete string appeared in the GloVe vocabulary.

| Tokenization | Token coverage | Type coverage |
|---|---:|---:|
| Whitespace | 85.33% (4,903,655 / 5,746,555) | 24.43% (57,548 / 235,606) |
| Regex, contractions intact | 97.67% (5,686,831 / 5,822,255) | 74.14% (60,326 / 81,371) |
| Regex, negative contractions expanded | 98.25% (5,753,223 / 5,855,455) | 74.17% (60,331 / 81,339) |

The biggest gain came from splitting punctuation off words such as `movie.`. Expanding `didn't` to `did not` and similar contractions reduced unmatched occurrences by **33,192**, or **24.5%**, while increasing the number of covered types by a net **five**. In the middle condition, **21,045 unmatched types** accounted for **135,424 occurrences**, or **6.43 occurrences per type** on average. I kept contractions intact in the classifiers to use one consistent tokenizer. The expansion experiment measured coverage, and its effect on classification remains open.

[Coverage results](vocabulary_results.json) · [Tokenizer and coverage code](vocabulary.py)

## 3. I looked at what GloVe knows and tried to reconstruct missing vectors

### Cosine neighbours

I retrieved the **10 nearest neighbours** of seven hand-picked sentiment and film words in the full 100d vocabulary, using cosine similarity and excluding the query itself. Ten neighbours made the results easy to inspect. This was a qualitative probe.

| Query | Example neighbour | Cosine |
|---|---|---:|
| awful | horrible | 0.922 |
| boring | dull | 0.699 |
| screenplay | script | 0.750 |
| overrated | underrated | 0.809 |

The vectors capture useful semantic and film-domain associations. The last example shows the limitation for sentiment classification. Words can have opposite evaluation polarity while appearing in similar contexts.

### Why I used a masked mean with a denominator of n − k

My first question was what to do when GloVe has no entry for a token. If a review has `n` tokens and `k` GloVe misses, I can put a zero at each missing position for the sum, then divide by **`n - k`**, the number of vectors I actually observed. For a ten-token review with two misses, I sum the eight available vectors and divide by eight. This is the same as skipping the missing tokens. Dividing by ten instead would scale the result toward zero as the missing fraction grows. Here I wanted a mean of the available lexical evidence. If no token has a vector, the implementation returns a zero review vector. Padding is excluded as well.

This decision fixes how to combine available vectors. The next question was whether I could recover some information from the missing words themselves. I considered averaging their neighbouring words, but in MeanPool that would effectively upweight the same neighbours, and an approximation of `didn't` based on `like` would still lose the explicit negation. I did not run this approach. I considered multiple imputation too, but the first experiment needed a simple, repeatable way to compare one replacement against another. I therefore ran single-vector baselines and a spelling-based predictor.

### Reconstructing a missing word vector

Given a word with an existing GloVe vector, I tried to predict that vector from spelling alone. The input was a binary bag of overlapping **3-, 4- and 5-character fragments**, with word-boundary markers. I chose this range to capture short shared fragments and longer spelling patterns while keeping the model sparse. I did not compare character ranges.

I used ridge regression with **`alpha=1.0`, `solver="lsqr"`, `tol=1e-4`**. The L2 penalty limits large weights on spelling features, LSQR handles the sparse input, and the tolerance is a fixed numerical setting. I did not tune alpha. I split the **60,326** covered types **90/10 by word type** using seed **42**, so the **6,033** evaluation words never appeared as fitting targets. The split leaves **54,293** types for fitting.

| Replacement | RMSE ↓ | Mean cosine ↑ |
|---|---:|---:|
| Zero vector | 0.472727 | Undefined |
| Mean vector of fitting word types | 0.467613 | 0.1559 |
| Character ridge | **0.467404** | **0.3904** |

Ridge was much closer in direction than the mean vector, while their RMSEs were close. I refit ridge on all covered types and generated vectors for **21,045 actual missing types**. Later, on the Project 1 split, I refit ridge on **68,026 covered training types** and exported two 100d review representations: average only covered tokens, or fill missing tokens and average all tokens. The first divides by `n - k`, where `k` is the number of missing tokens. This setup produced review vectors for **34,707 training and 7,437 validation reviews**. I did not run a sentiment classification comparison using the ridge-completed review vectors. The two files record distinct pooling rules: masked MeanPool uses `sum(covered) / (n - k)`, and the completed representation uses `(sum(covered) + sum(predicted)) / n`. On the Project 1 split, **31,778 training reviews and 6,813 validation reviews** contained at least one miss. This motivated testing what the classifier could learn directly from those missing tokens.

[Semantic probes](final_glove_study/semantic_probes.json) · [Reconstruction results](embedding_output/embedding_results.json) · [Review vector manifest](meanpool_output/manifest.json)

## 4. I tried giving the classifier the spelling of missing words directly

Instead of requiring the model to guess a plausible GloVe vector before predicting sentiment, I asked whether the sentiment labels could teach a classifier which missing spellings matter. For example, if a missing word were `organ`, the sliding bigrams would be `or`, `rg`, `ga` and `an`. The feature vector marks each fragment seen anywhere among the review's missing tokens. It can transfer fragments to new words, while discarding the identity and position of the original word. I used a linear or nonlinear head to learn their effects jointly with the review representation.

I kept the masked MeanPool vector and added a sparse binary vector of overlapping **character 2-grams from GloVe-missing words only**. Bigrams let short missing words share features. Binary encoding records whether a fragment appeared and keeps repetitions from increasing its value. The character vocabulary and MeanPool scaler were fitted on the training portion only.

The MLPs used **ReLU, Adam, `alpha=1e-4` and a 100-epoch cap**, with the scikit-learn defaults of initial learning rate 0.001, automatic batch size (200 here), Adam betas 0.9/0.999 and epsilon 1e-8. Those optimiser defaults were fixed so the feature and depth contrasts could be read separately. I started with **one 16-unit hidden layer** as a small nonlinear classification head. I added a second 16-unit layer to test depth without widening the hidden layers. These were fixed experiments, with seeds **42–44**.

| Experiment | Masked MeanPool | MeanPool + missing-word bigrams |
|---|---:|---:|
| 16-unit MLP, trained to epoch 100 | **80.51%** | 77.38% |
| 16 → 16 MLP, trained to epoch 100 | **79.78%** | 75.62% |
| 16 → 16 MLP, inner-validation checkpoint | **80.08%** | 80.04% |
| L2 logistic regression | **79.68%** | 79.56% |

The first two experiments hit the iteration cap. In the two-layer comparison, adding bigrams raised training Macro F1 from **84.68% to 93.68%** while validation fell. My first explanation was that 16 units might be too few to use both types of input. The deeper model's training score suggested it had enough capacity to fit the data, so I checked when the validation performance began to fall. I reran the experiment with per-epoch metrics on the inner split, using up to 100 epochs to observe what happens after the peak. The character model peaked at epochs **10, 6 and 12** (seeds 42–44); pure MeanPool peaked at **32, 46 and 45**. At epoch 100 that same diagnostic run was down to **75.00%** on the project validation set. Early checkpoint selection recovered most of the gap and left the two models nearly tied.

I then used L2 logistic regression to test the additional signal with a simpler head. I standardised the dense block on the fitting data and tried **`C = 0.001, 0.01, 0.1, 1, 10`** on inner validation. The log-spaced grid spans strong to weak regularisation. Both variants selected **0.1**. I used L-BFGS, an intercept, tolerance **1e-4** and a **1,000-iteration** cap. The tolerance was the library default and the cap gave the runs room to converge. Adding bigrams changed individual decisions: **187** reviews became correct and **196** became incorrect.

I also counted document frequency. **658 of 1,397** fitting-set fragments occurred in at most five reviews. That threshold was descriptive. I did not filter and rerun the model. Adding back the 3,471 internal-validation reviews raised the median fragment document frequency only from **6 to 7**, and only **899 of 31,236** fitting reviews contained any of the rare fragments. Those counts suggested sparsity in the feature space, but could not establish that rare fragments caused the performance drop. I calculated the odds of a positive label for reviews with and without each fragment, counting each review once. For example, `'t` had an odds ratio of **0.57** (approximate 95% interval **0.55–0.60**). I added 0.5 to all four cells only for tables with a zero cell. Some fragments were associated with sentiment, but the classification experiments found no consistent gain from adding them.

[One-layer MLP](oov_bigram_results.json) · [Two-layer MLP](oov_bigram_16x16_results.json) · [Learning curves](oov_learning_curves.png) · [Logistic comparison](oov_logistic_results.json) · [Odds ratios](oov_bigram_odds_ratios.csv)

## 5. I checked initialisation and mean versus max pooling

I compared random starts and GloVe starts with the same 100d MeanPool classifier and a **128-token input cap**. The vocabulary came from the fitting reviews. Both methods shared the same initial values for words missing from GloVe. Random vectors came from a zero-mean normal distribution with the standard deviation of the available GloVe vectors. The resulting standard deviation was **0.47511**. This keeps the scale comparable. Words outside the fitting vocabulary were skipped.

Both arms used a **linear head, Adagrad, learning rate 0.05, batch size 256, at most 12 epochs and patience 3**. Adagrad supports the sparse embedding gradients. Adagrad used zero weight decay, zero initial accumulator and epsilon 1e-10. The rate, batch and epoch budget were fixed for the comparison and were not independently tuned. I chose checkpoints by inner-validation Macro F1.

| MeanPool embedding treatment | Random start | GloVe start | GloVe advantage |
|---|---:|---:|---:|
| Frozen | 63.57% | **71.48%** | +7.91 pp |
| Trainable | 85.36% | **85.51%** | +0.15 pp |

Frozen GloVe won all three runs. Some of those heads were still improving at the epoch cap, so I also checked the frozen comparison with standardised features and L2 logistic regression, using the same `C` grid from Section 4. Across seeds the mean was **64.93% for frozen random** and **75.01% for frozen GloVe**. The chosen `C` was **0.1** for random and **1.0, 0.1, 0.01** for GloVe. The ordering survived the optimiser change. With trainable embeddings, GloVe won twice and the difference was small.

| Seed | Frozen random | Frozen GloVe | Trainable random | Trainable GloVe |
|---|---:|---:|---:|---:|
| 42 | 63.08% | 70.99% | 85.28% | 85.66% |
| 43 | 64.38% | 71.47% | 85.42% | 85.38% |
| 44 | 63.26% | 71.97% | 85.37% | 85.49% |

On the trainable runs, random starts actually led on inner validation after epoch 1 for every seed. Both approaches then settled near 87% on that inner split while training loss kept dropping. That is why I used the selected validation checkpoints and reported both the average and the per-seed results.

I then kept the **trainable GloVe settings and seed 42** and switched the aggregation from mean to per-dimension max. Max pooling ignored padding. **MeanPool scored 85.66% and MaxPool 83.58%** on project validation. I ran this comparison once and used mean pooling for the later comparisons.

[Trainable starts](initialization_results.json) · [Frozen starts](frozen_initialization_results.json) · [Frozen logistic check](final_glove_study/frozen_logistic_check.json) · [Mean vs. max](mean_vs_max_pooling_results.json) · [Learning curves](initialization_learning_curves.png)

## 6. I checked vector dimension and tuned the BiGRU

### GloVe dimension

I compared all four **GloVe 6B dimensions** using frozen MeanPool and L2 logistic regression. I used the same covered vocabulary, up to **256 GloVe-covered tokens**, fit the scaler on the fitting set, and tried the same `C` grid for every dimension. I did this on **inner validation**, to check whether the original 100d choice had a measurable cost.

| Dimension | Selected C | Inner-validation Macro F1 |
|---|---:|---:|
| 50d | 10 | 76.15% |
| 100d | 0.01 | 80.35% |
| 200d | 0.01 | 83.12% |
| 300d | 0.1 | **83.95%** |

More dimensions helped in this frozen pooling experiment. Higher dimensions also increase the number of linear classifier parameters. I kept 100d for the recurrent comparisons to maintain the earlier control, so this result does not tell us which dimension would work best with a GRU.

### Recurrent pilot and hyperparameter search

The first frozen **BiGRU** used a single bidirectional layer with **64 hidden units per direction**, and concatenated the two final states for a linear head. It can use context from both directions in an offline classification task. Using one layer isolated the move from pooling to recurrence without adding a depth comparison.

At the initial cap of **128 covered tokens**, seed 42 gave **83.86%** validation Macro F1 for BiGRU. The Adam MeanPool head was still improving after eight epochs and scored **70.01%**. I checked a better-optimised linear baseline on the same input. Standardised MeanPool + L2 logistic regression selected `C=0.01` on inner validation and reached **75.50%**.

I then screened frozen BiGRU configurations for **up to four epochs** on the same inner validation set, changing one setting at a time. The short budget kept the screen practical and can favour configurations that learn faster.

| Change from 64 units, 128 tokens, Adam 0.001 | Inner-validation Macro F1 |
|---|---:|
| Reference | 84.37% |
| Half the width, 32 units per direction | 83.81% |
| Half the learning rate, 0.0005 | 82.83% |
| Double the learning rate, 0.002 | 85.45% |
| Double the sequence cap, 256 covered tokens | **87.87%** |

I selected **64 units per direction, 256 tokens and Adam 0.001** because that tested combination had the best inner-validation score. The longer cap cut the share of truncated fitting reviews from **71.32% to 27.51%**. I did not test 0.002 at length 256. **Batch size 128**, global gradient norm clipping at **1.0**, **eight maximum epochs** for the confirmation runs, and **patience 2** were held over from the pilot. Clipping limits very large updates, and patience allows a short validation dip. These settings were controls and budget choices. I did not compare them against alternative values.

[Pilot](gru_comparison.json) · [Pilot linear check](gru_meanpool_logistic_check.json) · [Dimension screen](final_glove_study/dimension_screen.json) · [Recurrent search and full runs](final_glove_study/recurrent_results.json)

## 7. I compared the recurrent models against stronger pooling baselines

For the final comparison, I used GloVe 6B 100d and input sequences of up to the first **256 GloVe-covered tokens**. Missing tokens were skipped, and each model read the same retained tokens. The frozen MeanPool logistic regression selected **`C=0.01`** using the inner set. To avoid comparing a tuned recurrent system against only a frozen pooling baseline, I also ran trainable MeanPool using the same inputs and its earlier Adagrad settings.

I compared frozen BiGRU on **15,618** stratified fitting reviews with frozen BiGRU on all **31,236** fitting reviews. I used half the fitting set as a simple lower-data condition. With the same epoch budget, it also received fewer optimiser updates. I compared frozen and fine-tuned BiGRU on the full fitting set with the same Adam learning rate. Adam used its default betas (0.9, 0.999), epsilon (1e-8) and zero weight decay. I kept the defaults fixed across the recurrent ablations. Finally, I ran **one BiLSTM** with the BiGRU's width, input and training rules to compare the recurrent cells. The cells differ in parameter count at equal width. I did not tune BiLSTM independently.

| Model | Project validation Macro F1 | Project validation accuracy | Runs |
|---|---:|---:|---:|
| Frozen MeanPool + L2 logistic regression | 78.40% | 78.41% | 1 |
| Frozen BiGRU, half data | 85.57% ± 0.22 pp | 85.60% ± 0.20 pp | 3 |
| Frozen BiGRU, full data | 86.87% ± 0.36 pp | 86.87% ± 0.36 pp | 3 |
| Fine-tuned BiGRU, full data | 88.34% ± 0.23 pp | 88.36% ± 0.22 pp | 3 |
| Fine-tuned MeanPool, full data | **88.77% ± 0.12 pp** | **88.77% ± 0.12 pp** | 3 |
| Fine-tuned BiLSTM, full data | 86.86% | 86.88% | 1 |

The standard deviations above are across seeds. Frozen BiGRU beat frozen pooling by **8.47 pp**. Moving from half to full fitting data helped by **1.30 pp**, and fine-tuning the full-data GRU helped by **1.47 pp**. All three fine-tuned GRUs peaked at **epoch 2**. Their online training Macro F1 approached **98%** by epoch 4 while inner-validation Macro F1 declined, which is evidence of overfitting under these settings.

Fine-tuned MeanPool finished **0.43 pp** above fine-tuned BiGRU on average. They used different optimisers, so this compares the two fitted pipelines. The single-seed BiLSTM scored below both. I retained it as a comparison because both a GRU and an LSTM can capture order, but their gating differs. At the same 64-unit width, the BiLSTM had 7,282,521 trainable parameters versus 7,261,273 for the BiGRU, including embeddings. The comparison does not separate cell design from that difference in capacity. Project 1's historical **BPE token counts + SVM** scored **82.29%** on the same validation split, and its best **BPE bigram + MLP** scored **90.12%**. My best GloVe model landed between them.

### Hold-out comparison, seed 42

| Model | Inner-selected epoch | Validation Macro F1 | Test Macro F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Fine-tuned MeanPool | 4 | **88.85%** | **89.11%** | **89.11%** |
| Fine-tuned BiGRU | 2 | 88.57% | 88.76% | 88.76% |
| Fine-tuned BiLSTM | 5 | 86.86% | 86.69% | 86.69% |

BiLSTM was originally added as a validation-only comparison. I subsequently reproduced its epoch-5 checkpoint, verified every validation prediction matched, and ran the held-out test without changing its parameters.

[Validation accuracy](final_glove_study/validation_accuracy.json) · [Paired summary](final_glove_study/paired_summary.json) · [GRU learning curves](final_glove_study/gru_learning_curves.png) · [BiLSTM validation](final_glove_study/bilstm_results.json) · [BiLSTM test](final_glove_study/bilstm_test_results.json)

## 8. I checked where the final models disagreed

I paired the seed-42 predictions on the **same validation reviews**. Review-length cutoffs were the fitting-set tertiles at **139 and 231 tokens**, and the missing-word cutoff was the fitting-set median **2.10%**. I also marked whether a review was truncated or contained a negation or contrast cue. The cue flags use the full review, so the word triggering a flag may have been beyond the model's cutoff.

| Validation subset | Reviews | MeanPool Macro F1 | BiGRU Macro F1 | BiLSTM Macro F1 |
|---|---:|---:|---:|---:|
| Short | 2,468 | 89.57% | **90.39%** | 88.38% |
| Long | 2,455 | **86.64%** | 85.74% | 84.66% |
| Negation cue present | 6,510 | **88.59%** | 88.03% | 86.42% |
| Truncated | 2,031 | **86.15%** | 85.62% | 84.29% |

There were **286 reviews that only BiGRU got right** and **307 that only MeanPool got right**. The paired bootstrap over reviews used **10,000 resamples and seed 42**. It gave a 95% interval of **−0.94 to +0.35 pp** for BiGRU minus MeanPool Macro F1 at this fixed seed. That interval describes variation across reviews for those checkpoints.

I read three short reviews from each disagreement direction. In one, the reviewer praised the Canadian scenery but called the story boring. BiGRU correctly classified it as negative. In another, the reviewer criticised the plot but repeatedly said the movie was funny. MeanPool correctly classified it as positive. The two examples illustrate how mixed sentiment can lead to different errors.

I would next compare a **256-token prefix** against the same budget split between the beginning and end. Both models scored lower on truncated reviews, so retaining the ending is a concrete change to test. I would also separately test whether expanding negation helps classification. Neither follow-up has been run.

[Three-model predictions](final_glove_study/bilstm_validation_predictions.csv) · [All subgroup scores](final_glove_study/bilstm_results.json)

## Reproduce the experiments

After cloning the repository, run these commands from the directory containing this README and `vocabulary.py`. Set `PROJECT1` to the path of Project 1's `project_starter/experiment.py` in your checkout. The summarisation script also reads Project 1's historical `results/baseline/validation_scores.csv`. Exact dependency versions are recorded in [environment.json](final_glove_study/environment.json).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r gru_requirements.txt
curl -fL --retry 3 https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip -o glove.6B.zip
PROJECT1=/path/to/NLP1-Assignment1/project_starter/experiment.py

python vocabulary.py --glove glove.6B.zip
python complete_embeddings.py --glove glove.6B.zip
python prepare_meanpool.py --project1 "$PROJECT1" --glove glove.6B.zip
python compare_oov_bigrams.py --project1 "$PROJECT1" --glove glove.6B.zip
python compare_oov_bigrams.py --project1 "$PROJECT1" --glove glove.6B.zip --hidden-layers 16 16 --output oov_bigram_16x16_results.json
python diagnose_oov_learning.py --project1 "$PROJECT1" --glove glove.6B.zip
python compare_oov_logistic.py --project1 "$PROJECT1" --glove glove.6B.zip
python compare_initialization.py --project1 "$PROJECT1" --glove glove.6B.zip
python compare_initialization.py --project1 "$PROJECT1" --glove glove.6B.zip --freeze
python compare_gru.py --project1 "$PROJECT1" --glove glove.6B.zip
python finish_glove_study.py --project1 "$PROJECT1" --glove glove.6B.zip --semantics-only
python compare_glove_dimensions.py --project1 "$PROJECT1" --glove glove.6B.zip --limit 256
python finish_glove_study.py --project1 "$PROJECT1" --glove glove.6B.zip
python matched_trainable_meanpool.py --project1 "$PROJECT1" --glove glove.6B.zip
python matched_trainable_meanpool.py --project1 "$PROJECT1" --glove glove.6B.zip --save-predictions --evaluate-test
python summarize_glove_study.py --project1 "$PROJECT1"
python compare_bilstm.py --project1 "$PROJECT1" --glove glove.6B.zip
```

Pass `--arrow-dir` to the Project 1 split scripts if you already have the cached Arrow files. Use a working copy of the directory to preserve original results when rerunning. The global-mean imputation, odds ratios, original 128-token logistic check, max pooling comparison, frozen-logistic check, and BiLSTM held-out check were run as separate analysis commands. Their results are included here. The command list above covers the saved standalone scripts.

## Repository layout

| Path | Contents |
|---|---|
| `*.py` | Analysis and training code for the experiments above |
| `*requirements.txt` | Python dependencies for each stage |
| `embedding_output/` | Held-out reconstruction results and predicted GloVe vectors for missing word types |
| `meanpool_output/` | Masked and ridge-completed review vectors, labels and split metadata |
| `final_glove_study/` | Hyperparameter trials, validation and test metrics, paired predictions and error analysis |
| `*.json`, `*.csv`, `*.png` | Exploratory results and figures linked from the relevant sections |

The workflow figure predates the BiLSTM run. This README and the result files include the later BiLSTM evaluation.

Download GloVe 6B and create the virtual environment locally using the commands above. Project 1's split code and historical validation results are inputs to the reproduction commands.

Method references: [GloVe](https://aclanthology.org/D14-1162/), [GRU](https://doi.org/10.48550/arXiv.1406.1078), and [gradient clipping for recurrent networks](https://proceedings.mlr.press/v28/pascanu13.html).
