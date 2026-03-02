import os
import subprocess
import sys
import shutil

def run_cmd(cmd, cwd=None):
    print(f"🚀 Running: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        sys.exit(1)

def setup_third_party():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    third_party_dir = os.path.join(base_dir, "third_party")
    
    if not os.path.exists(third_party_dir):
        os.makedirs(third_party_dir)
        print(f"📂 Created {third_party_dir}")

    # 1. OpenCompass Setup
    oc_dir = os.path.join(third_party_dir, "opencompass")
    if not os.path.exists(oc_dir):
        print("📦 Cloning OpenCompass...")
        run_cmd("git clone https://github.com/open-compass/opencompass.git", cwd=third_party_dir)
        # Optional: Checkout specific commit if needed for stability
        # run_cmd("git checkout <commit-hash>", cwd=oc_dir)
    else:
        print("✅ OpenCompass already exists.")
    
    # Apply OpenCompass Patch
    patch_file = os.path.join(base_dir, "patches", "opencompass.patch")
    if os.path.exists(patch_file):
        print("🔧 Applying OpenCompass custom patches...")
        try:
            # Check if patch is already applied to avoid errors
            # Simple check: try dry-run
            run_cmd(f"git apply --check {patch_file}", cwd=oc_dir)
            run_cmd(f"git apply {patch_file}", cwd=oc_dir)
            print("✅ Patch applied successfully.")
        except:
            print("⚠️  Patch might already be applied or failed. Skipping.")
    
    print("🔧 Installing OpenCompass dependencies...")
    run_cmd(f"{sys.executable} -m pip install -e .", cwd=oc_dir)

    # 2. EvoPrompt Setup
    ep_dir = os.path.join(third_party_dir, "EvoPrompt")
    if not os.path.exists(ep_dir):
        print("📦 Cloning EvoPrompt...")
        run_cmd("git clone https://github.com/naszilla/EvoPrompt.git", cwd=third_party_dir)
    else:
        print("✅ EvoPrompt already exists.")

    # Apply EvoPrompt Patch
    ep_patch_file = os.path.join(base_dir, "patches", "evoprompt.patch")
    if os.path.exists(ep_patch_file):
        print("🔧 Applying EvoPrompt custom patches...")
        try:
            run_cmd(f"git apply --check {ep_patch_file}", cwd=ep_dir)
            run_cmd(f"git apply {ep_patch_file}", cwd=ep_dir)
            print("✅ Patch applied successfully.")
        except:
            print("⚠️  Patch might already be applied or failed. Skipping.")
            
    print("🔧 Installing EvoPrompt dependencies...")
    if os.path.exists(os.path.join(ep_dir, "requirements.txt")):
        run_cmd(f"{sys.executable} -m pip install -r requirements.txt", cwd=ep_dir)

    # 3. Core Dependencies
    print("📦 Installing CoT-Pilot core dependencies...")
    run_cmd(f"{sys.executable} -m pip install -r requirements.txt", cwd=base_dir)

    # 4. Dataset Download (Optional helper)
    # OpenCompass usually handles this, but we can ensure data/ folder exists
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        # Note: GSM8K is usually downloaded automatically by HuggingFace datasets or OpenCompass
        
    print("\n✨ Environment Setup Complete!")
    print("You can now run experiments using:")
    print("  python main.py --dataset gsm8k --model qwen3:0.6b")

if __name__ == "__main__":
    setup_third_party()
