
# ETAPA 2 - PASO 3: Normalización y Validación LOSO (Leave-One-Subject-Out)
# Funcionalidad:
#   - Lee `features_raw.csv`
#   - Ejecuta validación cruzada tipo LOSO: en cada fold se deja 1 sujeto para test.
#   - Normalización dentro del loop (opción segura por defecto: GLOBAL con media/std de TRAIN):
#       * global:     z-score con media/std calculadas en X_train (aplicadas a train y test).
#       * none:       sin normalización (usa valores crudos).
#       * transductive_subject: normaliza por SUJETO usando stats del sujeto de TEST
#   - Entrena RandomForest (opcional class_weight="balanced").
#   - Guarda métricas por sujeto (accuracy, macro-F1) y matriz de confusión agregada.
#
# Uso:
#   python 03_Normalizacion_Evaluacion_LOSO.py --input features_raw.csv --norm global --balanced yes
#
# Salidas:
#   - loso_results.csv         (accuracy y macro-F1 por sujeto)
#   - loso_confusion_matrix.png (matriz agregada)
#   - loso_report.txt          (resumen global)


import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def zscore_global_fit_transform(X_train):
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xn = (X_train - mu) / sd
    return Xn, mu, sd

def zscore_global_transform(X, mu, sd):
    return (X - mu) / sd

def zscore_per_subject(X, groups):
    """
    Normaliza por sujeto usando sus propias estadísticas (transductivo).
    groups: array-like con el nombre/ID del sujeto para cada fila de X.
    Devuelve Xn (mismo shape).
    """
    Xn = np.empty_like(X, dtype=float)
    for g in np.unique(groups):
        idx = (groups == g)
        Xi = X[idx]
        mu = np.nanmean(Xi, axis=0)
        sd = np.nanstd(Xi, axis=0, ddof=0)
        sd[sd == 0] = 1.0
        Xn[idx] = (Xi - mu) / sd
    return Xn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='features_raw.csv', help='CSV con features por ventanas')
    parser.add_argument('--norm', type=str, default='global', choices=['global','none','transductive_subject'],
                        help='Esquema de normalización dentro de cada fold')
    parser.add_argument('--balanced', type=str, default='yes', choices=['yes','no'],
                        help='Usar class_weight="balanced" en RandomForest')
    parser.add_argument('--n_estimators', type=int, default=300, help='Árboles del RandomForest')
    parser.add_argument('--random_state', type=int, default=42, help='Semilla')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Columnas
    feature_cols = [c for c in df.columns if c not in ('subject','label')]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df['label'].to_numpy()
    groups = df['subject'].to_numpy()

    logo = LeaveOneGroupOut()

    # Config del modelo
    cw = 'balanced' if args.balanced == 'yes' else None
    results = []
    cm_total = np.zeros((3,3), dtype=int)  # etiquetas 1,2,3

    all_y_true = []
    all_y_pred = []
    fold_idx = 0

    for train_idx, test_idx in logo.split(X, y, groups):
        fold_idx += 1
        subj_test = np.unique(groups[test_idx])
        assert len(subj_test) == 1, "Cada fold LOSO debe dejar 1 sujeto para test"
        subj_test = subj_test[0]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = groups[train_idx], groups[test_idx]

        # Normalización dentro del fold
        if args.norm == 'global':
            X_train_n, mu, sd = zscore_global_fit_transform(X_train)
            X_test_n = zscore_global_transform(X_test, mu, sd)
        elif args.norm == 'none':
            X_train_n, X_test_n = X_train, X_test
        elif args.norm == 'transductive_subject':
            # Usa estadísticas del propio sujeto de test (optimista; NO estricto)
            X_train_n = zscore_per_subject(X_train, g_train)
            X_test_n  = zscore_per_subject(X_test,  g_test)
        else:
            raise ValueError("norm no soportado")

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight=cw,
            n_jobs=-1
        )
        clf.fit(X_train_n, y_train)
        y_pred = clf.predict(X_test_n)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro')

        # CM por fold para etiquetas [1,2,3]
        labels_sorted = [1,2,3]
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_total += cm

        results.append({
            'subject': subj_test,
            'n_test_samples': len(y_test),
            'accuracy': acc,
            'f1_macro': f1m
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"[Fold {fold_idx:02d}] Test={subj_test} | acc={acc:.3f}, f1_macro={f1m:.3f}")

    # Guardar resultados
    res_df = pd.DataFrame(results).sort_values('subject').reset_index(drop=True)
    res_df.to_csv('loso_results.csv', index=False)
    print("\nResultados por sujeto guardados en loso_results.csv")
    print(res_df)

    # Reporte global
    report = classification_report(all_y_true, all_y_pred, labels=[1,2,3], digits=4)
    with open('loso_report.txt', 'w', encoding='utf-8') as f:
        f.write("Clasification report (agregado LOSO)\n")
        f.write(report + "\n")
        f.write("\nPromedios por sujeto:\n")
        f.write(res_df[['accuracy','f1_macro']].mean().to_string())

    print("\nReporte agregado (ver loso_report.txt):")
    print(report)

    # Matriz de confusión agregada
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm_total)
    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels([1,2,3])
    ax.set_yticklabels([1,2,3])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión agregada (LOSO)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm_total[i, j], ha='center', va='center')
    fig.tight_layout()
    fig.savefig('loso_confusion_matrix.png', dpi=160)
    print("Guardado: loso_confusion_matrix.png")

if __name__ == '__main__':
    main()
