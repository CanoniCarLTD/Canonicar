# compare_encoders.py
import argparse, importlib.util, torch, numpy as np, cv2, sys, os

def import_class_from_file(py_path, class_name):
    spec = importlib.util.spec_from_file_location("mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return getattr(mod, class_name)
    except AttributeError:
        raise RuntimeError(f"Class '{class_name}' not found in {py_path}")

def load_weights(model, weights_path, device):
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip leading "module." if present
    if isinstance(sd, dict):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")
    else:
        # last resort: if the class has a custom .load()
        if hasattr(model, "load") and callable(getattr(model, "load")):
            print("[INFO] Falling back to model.load()")
            model.load()
        else:
            raise RuntimeError("Weights format not recognized and model has no .load()")

def normalize(img_bgr, mode):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32)
    if mode == "none":
        pass
    elif mode == "01":
        x /= 255.0
    elif mode == "n11":
        x = (x / 255.0) * 2.0 - 1.0
    elif mode == "imagenet":
        x /= 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    # CHW
    return np.transpose(x, (2, 0, 1))

@torch.no_grad()
def run_once(model, tensor, device):
    model.eval().to(device)
    out = model(tensor.to(device))
    # Try to pick μ if a VAE returns (z, mu, logvar)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        # heuristic: choose the second as mu
        latent = out[1]
    else:
        latent = out
    latent = latent.detach().float().view(-1)
    lat_np = latent.cpu().numpy()
    return lat_np

def describe(name, vec):
    m, s = float(vec.mean()), float(vec.std())
    print(f"{name}: shape={tuple(vec.shape)}, mean={m:.6f}, std={s:.6f}, "
          f"min={float(vec.min()):.6f}, max={float(vec.max()):.6f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image (semantic or RGB frame)")
    ap.add_argument("--resize", nargs=2, type=int, default=[160, 80], help="W H target (default 160 80)")
    ap.add_argument("--norm", choices=["none","01","n11","imagenet"], default="01", help="Input normalization")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Encoder A (yours)
    ap.add_argument("--encA_file", required=True, help="Path to encoder .py (your encoder)")
    ap.add_argument("--encA_class", required=True, help="Class name (e.g., VariationalEncoder)")
    ap.add_argument("--encA_weights", required=True, help="Path to weights .pth for A")

    # Encoder B (reference / Idrees)
    ap.add_argument("--encB_file", required=True, help="Path to encoder .py (reference)")
    ap.add_argument("--encB_class", required=True, help="Class name")
    ap.add_argument("--encB_weights", required=True, help="Path to weights .pth for B")

    # Optional ctor kwargs
    ap.add_argument("--latent_dims", type=int, default=None, help="If your encoder needs latent size")
    ap.add_argument("--channels", type=int, default=3)
    args = ap.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        sys.exit(1)
    w, h = args.resize
    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    chw = normalize(img_bgr, args.norm)
    if chw.shape[0] != args.channels:
        print(f"[WARN] overriding channels to {args.channels}, current={chw.shape[0]}")
        # naive fix if someone uses 1-channel inputs
        if args.channels == 1:
            chw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if args.norm == "01":
                chw /= 255.0
            elif args.norm == "n11":
                chw = (chw / 255.0) * 2.0 - 1.0
            chw = chw[None, ...]
        else:
            # repeat or trim to fit
            if chw.shape[0] == 1 and args.channels == 3:
                chw = np.repeat(chw, 3, axis=0)
            else:
                chw = chw[:args.channels, ...]
    x = torch.from_numpy(chw[None, ...])  # [1,C,H,W]

    # Build A
    EncA = import_class_from_file(args.encA_file, args.encA_class)
    encA = EncA(args.latent_dims) if args.latent_dims is not None else EncA()
    load_weights(encA, args.encA_weights, args.device)

    # Build B
    EncB = import_class_from_file(args.encB_file, args.encB_class)
    encB = EncB(args.latent_dims) if args.latent_dims is not None else EncB()
    load_weights(encB, args.encB_weights, args.device)

    # Run
    latA = run_once(encA, x, args.device)
    latB = run_once(encB, x, args.device)

    # Describe
    describe("A (yours)", latA)
    describe("B (ref)",   latB)

    # Compare magnitude (rough)
    eps = 1e-12
    ratio_mean = (abs(latA.mean()) + eps) / (abs(latB.mean()) + eps)
    ratio_std  = (latA.std() + eps) / (latB.std() + eps)
    print(f"mean ratio A/B: {ratio_mean:.3f}   std ratio A/B: {ratio_std:.3f}")
    if ratio_std > 10.0 or ratio_std < 0.1:
        print("[ALERT] std differs by >10× — likely preprocessing/normalization mismatch.")

if __name__ == "__main__":
    main()
