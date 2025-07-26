import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Passo 3: Pré-processamento
def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}. Pulando.")
        return None
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

# Passo 4 e 5: Extração e Correspondência de Características
def match_fingerprints(img1_path, img2_path, detector_type='ORB'):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    if img1 is None or img2 is None:
        return 0, None

    kp1, des1 = None, None
    kp2, des2 = None, None

    if detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif detector_type == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("detector_type deve ser 'ORB' ou 'SIFT'")

    if des1 is None or len(des1) == 0 or des2 is None or len(des2) == 0:
        return 0, None

    matches = []
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print(f"Erro KNN match: {e}. Des1 shape: {des1.shape if des1 is not None else 'None'}, Des2 shape: {des2.shape if des2 is not None else 'None'}")
        return 0, None


    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: # Teste de razão de Lowe
            good_matches.append(m)

    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img

# Passo 6: Processamento do Dataset e Avaliação
def evaluate_matching_system(dataset_path, results_folder, detector_type='ORB', threshold=20):
    y_true = []
    y_pred = []
    all_match_counts = [] # Para a curva ROC

    os.makedirs(results_folder, exist_ok=True)
    print(f"\n--- Avaliando com {detector_type} e Threshold={threshold} ---")

    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(f"Aviso: Pulando '{folder_name}', esperado 2 imagens, encontrado {len(image_files)}.")
                continue

            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])

            match_count, match_img = match_fingerprints(img1_path, img2_path, detector_type)

            # Determina o ground truth com base no nome da pasta
            actual_match = 1 if "same" in folder_name.lower() else 0
            y_true.append(actual_match)
            all_match_counts.append(match_count) # Guarda o count para a ROC

            # Classificação
            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)

            result_str = "MATCHED" if predicted_match == 1 else "UNMATCHED"
            print(f"'{folder_name}': {result_str} ({match_count} good matches). Ground Truth: {actual_match}")

            # Salvar imagem de correspondência
            if match_img is not None:
                match_img_filename = f"{folder_name}_{result_str.lower()}_{detector_type}.png"
                match_img_path = os.path.join(results_folder, match_img_filename)
                cv2.imwrite(match_img_path, match_img)
                # print(f"  Imagem de correspondência salva em: {match_img_path}")

    # Passo 6: Avaliação de Desempenho
    if not y_true:
        print("Nenhuma comparação válida foi feita no dataset.")
        return

    # Matriz de Confusão
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix ({detector_type})")
    plt.show()

    # Curva ROC (apenas se houver ambos os rótulos 0 e 1 em y_true)
    if len(set(y_true)) > 1:
        fpr, tpr, _thresholds = roc_curve(y_true, all_match_counts)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve ({detector_type})')
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("Não há variação suficiente nos rótulos verdadeiros para plotar a curva ROC.")


# --- Exemplo de Uso ---
# Como 'data_check' está na mesma pasta que 'app.py', podemos referenciar diretamente:
data_check_path = "data_check" # Ou os.path.join(os.path.dirname(__file__), "data_check") para ser mais robusto
results_base_path = "." # As pastas de resultados serão criadas na raiz do projeto (FINGER-PRINT-APP)

# Experimentar com ORB
results_orb_folder = os.path.join(results_base_path, "orb_results")
evaluate_matching_system(data_check_path, results_orb_folder, detector_type='ORB', threshold=25) # Ajuste o threshold

# Experimentar com SIFT
results_sift_folder = os.path.join(results_base_path, "sift_results")
evaluate_matching_system(data_check_path, results_sift_folder, detector_type='SIFT', threshold=50) # Ajuste o threshold