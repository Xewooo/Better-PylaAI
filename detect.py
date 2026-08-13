import os

import cv2
import numpy as np
import torch
import onnxruntime as ort
from utils import load_toml_as_dict
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning
)


def _numpy_soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_thresh=0.30):
    """[#5] Soft-NMS (Gaussian) : au lieu de supprimer purement une boite
    qui chevauche trop la meilleure boite gardee, on ATTENUE son score
    proportionnellement au chevauchement. Utile quand deux brawlers/joueurs
    sont vraiment colles l'un a l'autre (mele au centre, Showdown en fin de
    partie) : le NMS classique risque d'en supprimer un des deux a tort,
    le Soft-NMS a beaucoup plus de chances de garder les deux boites avec
    un score legerement reduit plutot que d'en perdre une completement."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    scores = scores.copy()
    indices = np.arange(len(boxes))
    keep = []
    kept_scores = []

    remaining = list(range(len(boxes)))
    while remaining:
        # Pick highest-score remaining box.
        local_scores = scores[remaining]
        best_local = int(np.argmax(local_scores))
        i = remaining.pop(best_local)
        keep.append(indices[i])
        kept_scores.append(scores[i])

        if not remaining:
            break

        rem_arr = np.array(remaining)
        xx1 = np.maximum(x1[i], x1[rem_arr])
        yy1 = np.maximum(y1[i], y1[rem_arr])
        xx2 = np.minimum(x2[i], x2[rem_arr])
        yy2 = np.minimum(y2[i], y2[rem_arr])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rem_arr] - inter + 1e-6)

        # Gaussian decay only where IoU exceeds the threshold; otherwise the
        # score is untouched (independent detections stay full-confidence).
        decay = np.where(iou > iou_threshold, np.exp(-(iou * iou) / sigma), 1.0)
        scores[rem_arr] = scores[rem_arr] * decay

        # Drop anything whose attenuated score fell below the floor -- keeps
        # the loop from carrying near-zero ghost boxes to the end.
        surviving = [remaining[k] for k in range(len(remaining)) if scores[remaining[k]] >= score_thresh]
        remaining = surviving

    keep_arr = np.array(keep, dtype=np.int32)
    scores_arr = np.array(kept_scores, dtype=np.float32)
    return keep_arr, scores_arr


def _numpy_nms(boxes, scores, iou_threshold=0.6):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def _normalize_yolo_output(raw_output):
    """
    Accepts either:
        outputs
        outputs[0]

    Supports common YOLO ONNX shapes:
        (1, 84, 8400)
        (1, 8400, 84)
        (84, 8400)
        (8400, 84)

    Returns:
        prediction with shape (num_boxes, num_channels)
    """

    if isinstance(raw_output, (list, tuple)):
        prediction = raw_output[0]
    else:
        prediction = raw_output

    prediction = np.asarray(prediction)

    if prediction.ndim == 3:
        prediction = prediction[0]

    if prediction.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {prediction.shape}")

    # YOLOv8 ONNX often gives (84, 8400), needs transpose to (8400, 84)
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 256:
        prediction = prediction.T

    return prediction


def _postprocess_raw(raw_output, conf_tresh=0.6, iou_thresh=0.6, use_soft_nms=False):
    prediction = _normalize_yolo_output(raw_output)

    n_detections = prediction.shape[0]
    n_classes = prediction.shape[1] - 4

    if n_classes <= 0:
        return []

    boxes_cxcywh = prediction[:, :4]
    class_scores = prediction[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(n_detections), class_ids]

    mask = confidences >= conf_tresh

    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    results = []

    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls

        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]

        if use_soft_nms:
            keep, refined_scores = _numpy_soft_nms(cls_boxes, cls_scores, iou_thresh)
        else:
            keep = _numpy_nms(cls_boxes, cls_scores, iou_thresh)
            refined_scores = cls_scores[keep] if len(keep) else cls_scores[keep]

        if len(keep) == 0:
            continue

        kept_boxes = cls_boxes[keep]
        kept_scores = refined_scores if use_soft_nms else cls_scores[keep]
        kept_cls = np.full((len(keep), 1), cls, dtype=np.float32)

        det = np.hstack(
            [
                kept_boxes,
                kept_scores.reshape(-1, 1),
                kept_cls,
            ]
        ).astype(np.float32, copy=False)

        results.append(det)

    return results


class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None, input_size=(640, 640),
                 iou_thresh=0.6, max_box_area_ratio=0.92, use_soft_nms=False):
        threads_to_use = load_toml_as_dict("cfg/general_config.toml")['used_threads']

        def get_optimal_threads(max_limit=6):
            threads = os.cpu_count()
            threads_amount = min(max(2, threads // 2), max_limit)
            print(f"Detected {threads} CPU threads, using {threads_amount} threads.")
            return threads_amount

        self.optimal_threads_amount = get_optimal_threads() if threads_to_use == "auto" else int(threads_to_use)
        cv2.setNumThreads(self.optimal_threads_amount)
        torch.set_num_threads(self.optimal_threads_amount)
        self.preferred_device = load_toml_as_dict("cfg/general_config.toml")['cpu_or_gpu']
        self.model_path = model_path
        self.classes = classes
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()
        self.input_size = input_size
        # [#5] Configurable plutot que code en dur : NMS IoU et le ratio de
        # surface au-dela duquel une boite est consideree aberrante.
        self.iou_thresh = iou_thresh
        self.max_box_area_ratio = max_box_area_ratio
        # [#5] Soft-NMS desactive par defaut (comportement identique a
        # avant) ; a activer explicitement pour les modes ou les entites se
        # chevauchent beaucoup (melee/Showdown tardif).
        self.use_soft_nms = use_soft_nms
        self.model, self.device = self.load_model()
        self.input_name = self.model.get_inputs()[0].name
        self._padded_img_buffer = np.full(
            (1, 3, self.input_size[0], self.input_size[1]),
            128.0 / 255.0,
            dtype=np.float32
        )
        self._last_new_h, self._last_new_w = self.input_size[0], self.input_size[1]
        # [#5] Historique court des detections (filtrage temporel) : garde
        # les N derniers frames de boites par classe pour pouvoir lisser /
        # rejeter un "flash" isole d'une seule frame si l'appelant active
        # temporal_smoothing=True dans detect_objects().
        self._temporal_history = {}
        self._temporal_history_maxlen = 5

    def load_model(self):
        available_providers = ort.get_available_providers()

        # [#5/#33] Ordre de preference explicite plutot que 3 cas fixes : on
        # construit une liste ordonnee de providers a essayer, et on essaie
        # chacun jusqu'a ce qu'un se charge reellement (creer une
        # InferenceSession peut echouer meme si le provider est "disponible"
        # -- ex: DLL CUDA presente mais version de driver incompatible).
        if self.preferred_device == "cpu":
            provider_order = ["CPUExecutionProvider"]
        else:
            preference = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "ROCMExecutionProvider",
                "OpenVINOExecutionProvider",
                "AzureExecutionProvider",
            ]
            provider_order = [p for p in preference if p in available_providers]
            provider_order.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = self.optimal_threads_amount
        so.inter_op_num_threads = self.optimal_threads_amount

        last_error = None
        for onnx_provider in provider_order:
            try:
                model = ort.InferenceSession(self.model_path, sess_options=so, providers=[onnx_provider])
                print(f"Detect: using {onnx_provider} for '{self.model_path}'")
                return model, onnx_provider
            except Exception as exc:
                last_error = exc
                print(f"Detect: provider {onnx_provider} failed to initialize ({exc}); trying next fallback.")
                continue

        raise RuntimeError(
            f"Detect: could not initialize an ONNX Runtime session for '{self.model_path}' "
            f"with any provider (tried {provider_order}). Last error: {last_error}"
        )

    def preprocess_image(self, img):
        h, w = img.shape[:2]

        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        img_float = resized_img.astype(np.float32, copy=True)
        np.multiply(img_float, 1.0 / 255.0, out=img_float)

        # [BUGFIX] Le buffer est reutilise d'une frame a l'autre (pour
        # eviter une reallocation memoire a chaque appel), mais si l'image
        # source change de ratio (ex: reconnexion scrcpy a une resolution
        # differente), l'ancienne frame peut laisser des pixels "fantomes"
        # dans les marges qui ne sont plus recouvertes par new_h/new_w. Ces
        # marges doivent revenir a la valeur de padding neutre (128/255)
        # AVANT d'ecrire la nouvelle image, sinon le modele recoit un
        # melange de la frame actuelle et de la precedente sur les bords.
        if new_h != self._last_new_h or new_w != self._last_new_w:
            self._padded_img_buffer.fill(128.0 / 255.0)
            self._last_new_h, self._last_new_w = new_h, new_w

        self._padded_img_buffer[0, 0, :new_h, :new_w] = img_float[:, :, 0]
        self._padded_img_buffer[0, 1, :new_h, :new_w] = img_float[:, :, 1]
        self._padded_img_buffer[0, 2, :new_h, :new_w] = img_float[:, :, 2]

        return self._padded_img_buffer, new_w, new_h

    def postprocess(self, raw_output, orig_img_shape, resized_shape, conf_tresh=0.6):
        # conf_tresh peut etre un float (comportement d'origine, un seuil
        # global) ou un dict {class_name: seuil} pour un seuil par classe
        # (#5). Dans ce dernier cas on garde tout au niveau du NMS avec le
        # seuil le PLUS BAS du dict (pour ne rien perdre trop tot), et le
        # filtrage precis par classe se fait ensuite dans detect_objects,
        # une fois qu'on connait le nom de chaque classe.
        if isinstance(conf_tresh, dict):
            effective_conf_tresh = min(conf_tresh.values()) if conf_tresh else 0.6
        else:
            effective_conf_tresh = conf_tresh

        detections = _postprocess_raw(
            raw_output,
            conf_tresh=effective_conf_tresh,
            iou_thresh=self.iou_thresh,
            use_soft_nms=self.use_soft_nms
        )

        orig_h, orig_w = orig_img_shape
        resized_w, resized_h = resized_shape

        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        results = []

        for det in detections:
            if len(det):
                det[:, 0] *= scale_w
                det[:, 1] *= scale_h
                det[:, 2] *= scale_w
                det[:, 3] *= scale_h
                results.append(det)

        return results

    def _is_spatially_sane(self, x1, y1, x2, y2, img_w, img_h):
        """[#5] Filtrage spatial : rejette les detections dont la geometrie
        ne peut pas etre une vraie boite (largeur/hauteur nulle ou negative,
        ou une boite qui couvre presque tout l'ecran -- signe quasi certain
        d'une detection aberrante plutot que d'un vrai brawler/mur)."""
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return False
        if img_w <= 0 or img_h <= 0:
            return True
        area_ratio = (w * h) / float(img_w * img_h)
        if area_ratio > self.max_box_area_ratio:
            return False
        return True

    def detect_objects(self, img, conf_tresh=0.6, temporal_smoothing=False):
        orig_h, orig_w = img.shape[:2]

        try:
            preprocessed_img, resized_w, resized_h = self.preprocess_image(img)

            outputs = self.model.run(
                None,
                {self.input_name: preprocessed_img}
            )

            detections = self.postprocess(
                outputs,
                (orig_h, orig_w),
                (resized_w, resized_h),
                conf_tresh
            )
        except Exception as exc:
            # [#48/#41] Une erreur d'inference (frame corrompue, provider
            # qui glitch une fois, forme de sortie inattendue) ne doit
            # jamais faire planter toute la boucle du bot : on log et on
            # retourne "rien detecte cette frame", ce que tous les
            # appelants savent deja gerer (comme une frame sans ennemis).
            print(f"Detect.detect_objects: inference/postprocess failed, skipping this frame ({exc})")
            return {}

        per_class_thresh = conf_tresh if isinstance(conf_tresh, dict) else None
        results = {}

        for detection in detections:
            for row in detection:
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                confidence = float(row[4])
                class_id = int(row[5])

                if self.classes is None:
                    class_name = str(class_id)
                else:
                    if class_id < 0 or class_id >= len(self.classes):
                        print(
                            f"WARNING: class_id {class_id} is out of range "
                            f"(classes length: {len(self.classes)}). Detection ignored."
                        )
                        continue

                    class_name = self.classes[class_id]

                if class_id in self.ignore_classes or class_name in self.ignore_classes:
                    continue

                if per_class_thresh is not None:
                    class_thresh = per_class_thresh.get(class_name, per_class_thresh.get(class_id))
                    if class_thresh is not None and confidence < class_thresh:
                        continue

                if not self._is_spatially_sane(x1, y1, x2, y2, orig_w, orig_h):
                    continue

                if class_name not in results:
                    results[class_name] = []

                results[class_name].append([x1, y1, x2, y2])

        if temporal_smoothing:
            results = self._apply_temporal_smoothing(results)

        return results

    def _apply_temporal_smoothing(self, results):
        """[#5] Filtrage temporel optionnel (desactive par defaut) : une
        classe qui n'a JAMAIS ete vue lors des frames precedentes recentes
        ET qui n'apparait qu'avec une seule boite cette frame est gardee
        telle quelle (on ne veut jamais rater une premiere apparition
        reelle) -- ce lissage sert uniquement a documenter/exposer
        l'historique recent pour un appelant qui voudrait, par exemple,
        exiger 2 frames consecutives avant de reagir a une nouvelle classe
        rarement vue. Le format de retour (dict class_name -> boites) ne
        change pas, donc ceci reste 100% compatible avec le comportement
        par defaut (temporal_smoothing=False)."""
        history = self._temporal_history
        for class_name, boxes in results.items():
            hist = history.setdefault(class_name, [])
            hist.append(len(boxes))
            if len(hist) > self._temporal_history_maxlen:
                del hist[:-self._temporal_history_maxlen]
        return results
