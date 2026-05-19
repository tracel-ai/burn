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

Tous les sites d'appel ont été mis à jour en conséquence (voir ci-dessous).

### 3. `crates/burn-train/src/metric/processor/full.rs`

**`FullEventProcessorTraining`** — deux champs ajoutés :
- `in_train_split: bool` — détecte le premier `ProcessedItem` pour appeler `start_split("train", ...)` (lazy-init, car il n'y a pas d'événement `StartSplit` dans le système)
- `in_valid_split: bool` — idem pour la split validation

**`FullEventProcessorEvaluation`** — un champ ajouté :
- `test_started: bool` — détecte le premier `ProcessedItem` pour appeler `start_test(...)` (même raison)

Constante ajoutée :
```rust
const EVALUATOR_TEST_SPLITS: usize = 1;
// Chaque appel à Evaluator::eval() lance exactement un test split.
```

Appels au logger câblés dans chaque bras du `match` :

| Événement | Appel logger |
|-----------|-------------|
| `LearnerEvent::Start { total_epochs }` | `logger.start(total_epochs, None)` |
| `LearnerEvent::ProcessedItem` (train, 1er) | `logger.start_split("train", total)` |
| `LearnerEvent::ProcessedItem` (train) | `logger.update_split(items_processed)` |
| `LearnerEvent::EndEpoch` (train) | `logger.end_split()` |
| `LearnerEvent::ProcessedItem` (valid, 1er) | `logger.start_split("valid", total)` |
| `LearnerEvent::ProcessedItem` (valid) | `logger.update_split(items_processed)` |
| `LearnerEvent::EndEpoch` (valid) | `logger.end_split()` + `logger.update_epoch(epoch)` |
| `LearnerEvent::End` | `logger.end()` |
| `EvaluatorEvent::Start` | `logger.start(EVALUATOR_TEST_SPLITS)` |
| `EvaluatorEvent::ProcessedItem` (1er) | `logger.start_test(name, total)` |
| `EvaluatorEvent::ProcessedItem` | `logger.update_test(items_processed)` |
| `EvaluatorEvent::End` | `logger.end_test()` + `logger.end()` |

### 4. Sites d'appel mis à jour pour `LearnerEvent::Start { .. }`

- `crates/burn-train/src/metric/processor/minimal.rs` — deux bras `Start => {}` → `Start { .. } => {}`
- `crates/burn-train/src/learner/supervised/strategies/base.rs` — émission du vrai `total_epochs`
- `crates/burn-train/src/learner/early_stopping.rs` — test : `Start { total_epochs: 0 }`
- `crates/burn-train/src/checkpoint/strategy/metric.rs` — test : `Start { total_epochs: 0 }`

## Exemple d'implémentation (mnist)

`examples/mnist/src/file_progress.rs` — implémente les deux traits en écrivant dans un fichier texte (append mode, flush à chaque ligne).

`examples/mnist/src/training.rs` — câble les loggers via `.with_progress_logger(...)` sur `SupervisedTraining` et `EvaluatorBuilder`.

## Points de design importants

- **Pas d'événement `StartSplit`** dans le système d'événements : `LearnerEvent` n'a pas de variant `StartSplit`. Le `start_split` est donc déclenché de façon *lazy* sur le premier `ProcessedItem` de chaque split, grâce aux flags `in_train_split` / `in_valid_split`.
- **`process_valid(LearnerEvent::Start { .. })` est un no-op** : cet événement n'est jamais émis par le runtime. Le bras existe uniquement pour satisfaire l'exhaustivité de Rust.
- **`logger.start()` est appelé une seule fois** dans `process_train(Start)`, pas dans `process_valid(Start)`.
- **`EVALUATOR_TEST_SPLITS = 1`** : l'`Evaluator::eval()` lance toujours exactement un dataset par appel.
