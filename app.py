from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
import io
import os
import zipfile
import tempfile
import shutil
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

app = Flask(__name__)

CLASS_LABELS = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

CLASS_INFO = {
    'EOSINOPHIL': {
        'description': 'Combats parasites and modulates allergic responses. Elevated levels may indicate allergies, asthma, or parasitic infections.',
        'normal_range': '1–4% of WBC', 'color': '#FF6B6B', 'icon': '🔴'
    },
    'LYMPHOCYTE': {
        'description': 'Key players in adaptive immunity. B-cells produce antibodies; T-cells destroy infected cells. Central to immune memory.',
        'normal_range': '20–40% of WBC', 'color': '#4ECDC4', 'icon': '🟢'
    },
    'MONOCYTE': {
        'description': 'Phagocytic cells that engulf pathogens and debris. Differentiate into macrophages and dendritic cells in tissues.',
        'normal_range': '2–8% of WBC', 'color': '#FFE66D', 'icon': '🟡'
    },
    'NEUTROPHIL': {
        'description': 'First responders to bacterial infection. Most abundant WBC. Rapidly migrate to sites of infection and inflammation.',
        'normal_range': '55–70% of WBC', 'color': '#A8E6CF', 'icon': '🔵'
    }
}

MODEL_PATH = 'best_model.keras'
model = None


def _build_architecture():
    from tensorflow.keras import layers, models, regularizers
    m = models.Sequential([
        layers.SeparableConv2D(128, (8,8), strides=(3,3), activation='relu', input_shape=(224,224,3)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (5,5), strides=(1,1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(256, (3,3), strides=(1,1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(256, (1,1), strides=(1,1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(256, (1,1), strides=(1,1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.SeparableConv2D(512, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.SeparableConv2D(512, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.SeparableConv2D(512, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.SeparableConv2D(512, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    m(np.zeros((1, 224, 224, 3), dtype=np.float32))
    return m


def _assign_weights_from_h5(model, h5_path):
    """
    Keras 3 stores weights grouped by layer name under /layers/<name>/vars/0,1,2...
    The vars ordering within each layer matches the Keras 2 weight ordering.
    We must match by LAYER NAME, not just shape, to avoid scrambling.
    """
    with h5py.File(h5_path, 'r') as f:
        # Map: layer_name -> [array0, array1, ...] in vars order
        layer_weights = {}
        if 'layers' not in f:
            raise KeyError("Expected 'layers' group in h5 file")
        
        layers_group = f['layers']
        for layer_name in layers_group.keys():
            layer_group = layers_group[layer_name]
            if 'vars' not in layer_group:
                continue
            vars_group = layer_group['vars']
            # Sort by integer key: '0', '1', '2'...
            sorted_keys = sorted(vars_group.keys(), key=lambda x: int(x))
            arrays = [vars_group[k][()] for k in sorted_keys]
            layer_weights[layer_name] = arrays
            print(f"  [h5] {layer_name}: {[a.shape for a in arrays]}")

    print(f"\n[INFO] h5 layers found: {list(layer_weights.keys())}")

    # Build ordered list matching Keras 2 weight tensor order.
    # Keras 2 weight order per layer type:
    #   SeparableConv2D: depthwise_kernel, pointwise_kernel, bias
    #   BatchNormalization: gamma, beta, moving_mean, moving_variance
    #   Dense: kernel, bias
    # Keras 3 vars order is the SAME — so we just iterate model layers
    # and pull from the matching h5 layer by name.

    assignments = []
    model_weights = model.weights

    # Build a map from model layer name -> list of weight tensors
    layer_weight_map = {}
    for w in model_weights:
        # w.name like 'separable_conv2d/depthwise_kernel:0'
        layer_name = w.name.split('/')[0]
        layer_weight_map.setdefault(layer_name, []).append(w)

    print(f"\n[INFO] Model layers: {list(layer_weight_map.keys())}")

    # Iterate model weights in order, pull h5 arrays by layer name + position
    layer_position = {}  # track how many weights we've consumed per layer
    final_assignments = []

    for w in model_weights:
        layer_name = w.name.split('/')[0]  # e.g. 'separable_conv2d'
        pos = layer_position.get(layer_name, 0)
        layer_position[layer_name] = pos + 1

        if layer_name not in layer_weights:
            raise KeyError(f"Layer '{layer_name}' not found in h5. Available: {list(layer_weights.keys())}")

        available = layer_weights[layer_name]
        if pos >= len(available):
            raise IndexError(f"Layer '{layer_name}' has {len(available)} arrays but need index {pos}")

        arr = available[pos]
        if tuple(arr.shape) != tuple(w.shape):
            raise ValueError(
                f"Shape mismatch for '{w.name}': model={tuple(w.shape)}, h5={tuple(arr.shape)}"
            )
        
        final_assignments.append(arr)
        print(f"  OK  {w.name} {tuple(w.shape)} <- {layer_name}/vars/{pos}")

    model.set_weights(final_assignments)
    print(f"\n[INFO] ✅ All {len(final_assignments)} weights assigned correctly by layer name!")


def load_model_once():
    global model
    if model is not None:
        return model
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] {MODEL_PATH} not found — demo mode")
        return None

    # Strategy 1: standard load
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("[INFO] Strategy 1 OK: standard load_model")
        return model
    except Exception as e1:
        print(f"[WARN] Strategy 1 failed: {type(e1).__name__}: {e1}")

    # Strategy 2: extract model.weights.h5 from .keras ZIP, assign by layer name
    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
            names = zf.namelist()
            print(f"[INFO] .keras ZIP contents: {names}")
            wfile = next((n for n in names if 'weight' in n.lower() and n.endswith('.h5')), None)
            if wfile is None:
                raise FileNotFoundError(f"No weights .h5 found. Contents: {names}")
            extracted = zf.extract(wfile, tmpdir)

        m = _build_architecture()
        _assign_weights_from_h5(m, extracted)
        model = m
        print("[INFO] Strategy 2 OK: layer-name matched weight loading")
        return model
    except Exception as e2:
        print(f"[WARN] Strategy 2 failed: {type(e2).__name__}: {e2}")
        import traceback; traceback.print_exc()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("[ERROR] All strategies exhausted — demo mode")
    return None


def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image.convert('RGB')).astype(np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def generate_gradcam(model, img_array, pred_index):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'separable_conv' in layer.name or 'conv2d' in layer.name:
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
        hmax = heatmap.max()
        if hmax > 0:
            heatmap = heatmap / hmax
        heatmap = np.clip(heatmap, 0.0, 1.0)

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_uint8).resize((224, 224), Image.LANCZOS)
        heatmap_np = np.array(heatmap_img).astype(np.float32) / 255.0

        try:
            colormap = cm.colormaps['jet']
        except AttributeError:
            colormap = plt.get_cmap('jet')

        heatmap_colored = np.uint8(255 * colormap(heatmap_np)[:, :, :3])
        orig = np.uint8((img_array[0] + 1) / 2 * 255)
        superimposed = np.uint8(heatmap_colored * 0.4 + orig * 0.6)

        buf = io.BytesIO()
        Image.fromarray(superimposed).save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"GradCAM error: {e}")
        import traceback; traceback.print_exc()
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    img_array = preprocess_image(image)
    m = load_model_once()

    if m is None:
        probs = np.array([0.05, 0.12, 0.08, 0.75])
        pred_index = 3
        demo = True
    else:
        preds = m.predict(img_array, verbose=0)
        probs = preds[0]
        pred_index = int(np.argmax(probs))
        demo = False
        print(f"[PREDICT] probs={probs}  pred={CLASS_LABELS[pred_index]}")

    predicted_class = CLASS_LABELS[pred_index]
    confidence = float(probs[pred_index]) * 100
    all_probs = {CLASS_LABELS[i]: float(probs[i]) * 100 for i in range(4)}
    heatmap_b64 = generate_gradcam(m, img_array, pred_index) if m else None

    buf = io.BytesIO()
    image.resize((224, 224)).save(buf, format='PNG')
    buf.seek(0)
    orig_b64 = base64.b64encode(buf.read()).decode('utf-8')

    response = {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': all_probs,
        'heatmap': heatmap_b64,
        'original': orig_b64,
        'info': CLASS_INFO[predicted_class],
        'demo_mode': demo
    }
    print(f"[PREDICT] Returning: prediction={predicted_class}, confidence={confidence:.1f}%, heatmap={'YES' if heatmap_b64 else 'NO'}")
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)