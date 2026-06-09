import cv2
import numpy as np
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor


CLASS_LABELS = {
    0: "F16",
    1: "Helikopter",
    2: "Balistik_Fuze",
    3: "Mini_Micro_IHA",
    4: "Dost"
}


THREAT_CLASSES = {"F16", "Helikopter", "Balistik_Fuze", "Mini_Micro_IHA"}


class ViTClassifier:
    def __init__(self, model_path=None, confidence_threshold=0.75):
        
        model_path = r"C:\Users\saadz\OneDrive\Desktop\SARUT\Yazilim\vitmodel"
        
        self.threshold = confidence_threshold
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path:
            # Eğitilmiş model
            self.processor = ViTImageProcessor.from_pretrained(model_path)
            self.model     = ViTForImageClassification.from_pretrained(model_path)
        else:
            # Placeholder — model henüz eğitilmedi
            pretrained = "google/vit-base-patch16-224"
            self.processor = ViTImageProcessor.from_pretrained(pretrained)
            self.model     = ViTForImageClassification.from_pretrained(
                pretrained,
                num_labels=len(CLASS_LABELS),
                ignore_mismatched_sizes=True   # sınıf sayısı farklı, uyarıyı bastır
            )

        self.model.to(self.device)
        self.model.eval()

    def classify(self, frame, bbox):
        """
        frame : BGR numpy array (kamera karesi)
        bbox  : (x, y, w, h) — CSRT tracker çıktısı

        Döndürür:
            label      : str  — sınıf adı veya "belirsiz"
            confidence : float — [0, 1]
            is_threat  : bool
        """
        roi = self._crop_roi(frame, bbox)
        if roi is None:
            return "belirsiz", 0.0, False

        # BGR → RGB → PIL
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # ViT ön işleme: 224x224 yeniden boyutlandırma + normalizasyon
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs    = self.model(**inputs)
            logits     = outputs.logits                          # (1, num_classes)
            probs      = torch.softmax(logits, dim=-1)[0]        # (num_classes,)
            confidence = probs.max().item()
            class_idx  = probs.argmax().item()

        label     = CLASS_LABELS.get(class_idx, "belirsiz")
        is_threat = (label in THREAT_CLASSES) and (confidence >= self.threshold)

        # Eşik altı → belirsiz
        if confidence < self.threshold:
            label = "belirsiz"

        return label, confidence, is_threat

    def _crop_roi(self, frame, bbox):
        """CSRT bbox'ından ROI kırpar, geçersizse None döner."""
        x, y, w, h = [int(v) for v in bbox]
        fh, fw     = frame.shape[:2]

        # Sınır kontrolü
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)

        if w <= 0 or h <= 0:
            return None

        return frame[y:y+h, x:x+w]


# ── Bağımsız test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tracker import Tracker

    cap        = cv2.VideoCapture(0)
    tracker    = Tracker(profile="Red")
    classifier = ViTClassifier(model_path=None, confidence_threshold=0.75)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mode, bbox = tracker.update(frame)

        label, confidence, is_threat = "—", 0.0, False

        if mode == "tracking" and bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Her 10 karede bir sınıflandır (her karede çalıştırmak pahalı)
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
                label, confidence, is_threat = classifier.classify(frame, bbox)

            color = (0, 0, 255) if is_threat else (255, 0, 0)
            cv2.putText(frame, f"{label} ({confidence:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status_color = (0, 0, 255) if is_threat else (200, 200, 200)
        cv2.putText(frame, f"Mod: {mode} | Sinif: {label} | Tehdit: {is_threat}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("SARUT - ViT Siniflandirici", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()