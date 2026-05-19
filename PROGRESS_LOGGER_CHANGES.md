# Progress Logger — Résumé des changements

## Contexte

Cette branche ajoute deux traits de logging de progression (`TrainingProgressLogger` et `EvaluationProgressLogger`) dans `burn-train` et les câble dans le pipeline d'entraînement/évaluation existant.

## Fichiers modifiés

### 1. `crates/burn-train/src/logger/progress.rs` *(nouveau contenu)*

Définit les deux traits avec leur documentation et leur séquence d'appel attendue.

**`TrainingProgressLogger`** — séquence :
```
start(total_epochs, total_items?)
  pour chaque époque :
    start_split("train", total_items_train)
      update_split(items_processed)  // une fois par batch
    end_split()
    start_split("valid", total_items_valid)
      update_split(items_processed)
    end_split()
    update_epoch(epoch)
end()
```

**`EvaluationProgressLogger`** — séquence :
```
start(total_tests)
  pour chaque test split :
    start_test(name, total_items)
      update_test(items_processed)  // une fois par batch
    end_test()
end()
```

### 2. `crates/burn-train/src/metric/processor/base.rs`

`LearnerEvent::Start` est passé de variant unité à variant struct pour transporter `total_epochs` :

```rust
// Avant
LearnerEvent::Start

// Après
LearnerEvent::Start { total_epochs: usize }
```

`LearnerEvent::EndEpoch` est remplacé par deux nouveaux variants pour délimiter explicitement chaque split :

```rust
// Avant
LearnerEvent::EndEpoch(usize)

// Après
LearnerEvent::StartSplit(usize)   // total d'items dans le split
LearnerEvent::EndSplit(usize)     // numéro d'époque courant
```

`EvaluatorEvent::StartTest` est ajouté pour signaler explicitement le début d'un test split :

```rust
// Avant : pas d'événement StartTest — détection lazy sur le premier ProcessedItem
// Après
EvaluatorEvent::StartTest(EvaluationName, usize)  // nom + total d'items
```

Tous les sites d'appel ont été mis à jour en conséquence (voir ci-dessous).

### 3. `crates/burn-train/src/metric/processor/full.rs`

**`FullEventProcessorTraining`** — aucun champ booléen de suivi d'état : les événements `StartSplit` / `EndSplit` délimitent explicitement chaque split, rendant les flags `in_train_split` / `in_valid_split` inutiles.

**`FullEventProcessorEvaluation`** — aucun champ booléen de suivi d'état : le nouvel événement `StartTest` remplace le flag `test_started`.

Constante ajoutée :
```rust
const EVALUATOR_TEST_SPLITS: usize = 1;
// Chaque appel à Evaluator::eval() lance exactement un test split.
```

Appels au logger câblés dans chaque bras du `match` :

| Événement | Appel logger |
|-----------|-------------|
| `LearnerEvent::Start { total_epochs }` | `logger.start(total_epochs, None)` |
| `LearnerEvent::StartSplit` (train) | `logger.start_split("train", total_items)` |
| `LearnerEvent::ProcessedItem` (train) | `logger.update_split(items_processed)` |
| `LearnerEvent::EndSplit` (train) | `logger.end_split()` |
| `LearnerEvent::StartSplit` (valid) | `logger.start_split("valid", total_items)` |
| `LearnerEvent::ProcessedItem` (valid) | `logger.update_split(items_processed)` |
| `LearnerEvent::EndSplit` (valid) | `logger.end_split()` + `logger.update_epoch(epoch)` |
| `LearnerEvent::End` | `logger.end()` |
| `EvaluatorEvent::Start` | `logger.start(EVALUATOR_TEST_SPLITS)` |
| `EvaluatorEvent::StartTest(name, total_items)` | `logger.start_test(name, total_items)` |
| `EvaluatorEvent::ProcessedItem` | `logger.update_test(items_processed)` |
| `EvaluatorEvent::End` | `logger.end_test()` + `logger.end()` |

### 4. `crates/burn-train/src/evaluator/base.rs`

Émission du nouvel événement `StartTest` avant la boucle principale :

```rust
let total_items = dataloader.num_items();
self.event_processor.process_test(EvaluatorEvent::Start);
self.event_processor.process_test(EvaluatorEvent::StartTest(name.clone(), total_items));
// boucle sur les items...
```

### 5. Sites d'appel mis à jour pour `LearnerEvent::Start { .. }`

- `crates/burn-train/src/metric/processor/minimal.rs` — deux bras `Start => {}` → `Start { .. } => {}`
- `crates/burn-train/src/learner/supervised/strategies/base.rs` — émission du vrai `total_epochs`
- `crates/burn-train/src/learner/early_stopping.rs` — test : `Start { total_epochs: 0 }`
- `crates/burn-train/src/checkpoint/strategy/metric.rs` — test : `Start { total_epochs: 0 }`

## Exemple d'implémentation (mnist)

`examples/mnist/src/file_progress.rs` — implémente les deux traits en écrivant dans un fichier texte (append mode, flush à chaque ligne).

`examples/mnist/src/training.rs` — câble les loggers via `.with_progress_logger(...)` sur `SupervisedTraining` et `EvaluatorBuilder`.

## Points de design importants

- **Événements `StartSplit` / `EndSplit` explicites** : `LearnerEvent` dispose maintenant de variants dédiés pour délimiter chaque split. Le runtime les émet depuis la stratégie d'entraînement (`SingleDeviceTrainingStrategy`, etc.), ce qui élimine tout besoin de lazy-init dans le processor.
- **Événement `StartTest` explicite** : `EvaluatorEvent` dispose maintenant d'un variant `StartTest` émis par `Evaluator::eval()` avant la boucle, en passant `dataloader.num_items()`. Cela remplace le flag `test_started` qui déclenchait `start_test` de façon lazy sur le premier `ProcessedItem`.
- **`process_valid(LearnerEvent::Start { .. })` est un no-op** : cet événement n'est jamais émis par le runtime. Le bras existe uniquement pour satisfaire l'exhaustivité de Rust.
- **`logger.start()` est appelé une seule fois** dans `process_train(Start)`, pas dans `process_valid(Start)`.
- **`EVALUATOR_TEST_SPLITS = 1`** : l'`Evaluator::eval()` lance toujours exactement un dataset par appel.
