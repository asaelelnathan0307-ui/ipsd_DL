# %% [1] Cek GPU & pasang paket (jalankan SEKALI per sesi Colab)
import subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "kagglehub"])
print("Paket terpasang")

# %% [2] Kredensial Kaggle (isi sendiri, JANGAN di-commit ke GitHub)
import os
os.environ["KAGGLE_USERNAME"] = "ISI_USERNAME"
os.environ["KAGGLE_KEY"] = "ISI_KEY"

# %% [3] Unduh hasil (jalankan setelah test.py selesai; *.pt tidak ikut)
import base64, zipfile, glob
from IPython.display import HTML, display

with zipfile.ZipFile("hasil.zip", "w") as z:
    for f in glob.glob("Results/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".pt"):
            z.write(f)
with open("hasil.zip", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
display(HTML(f'<a download="hasil.zip" href="data:application/zip;base64,{b64}" style="font-size:16px;font-weight:bold;">KLIK UNTUK DOWNLOAD hasil.zip</a>'))
# %%
