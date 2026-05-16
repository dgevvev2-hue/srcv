# PylaAI

PylaAI is currently the best external Brawl Stars bot. This repository is intended for developers.

> ⚠️ **Warning:** This repository contains the **source code**.  
> If you are not a developer, it is recommended to use the **official compiled build** from our Discord (linked below), which comes as a ready-to-use `.exe`.

---

# Supported Platforms
- **Windows 10/11**

## Hardware Support


### Supported Hardware

- **NVIDIA GPUs**
  - Automatically installs compatible **CUDA + PyTorch**
  - Optimized for **GTX 10-series → RTX 50-series**

- **AMD GPUs**
  - Native **ROCm** support for Radeon / Ryzen GPUs

- **Intel / Generic GPUs**
  - Uses **DirectML** acceleration on Windows
  - Works well with integrated graphics

- **Linux / WSL**
  - Fully optimized for **Ubuntu / WSLg environments**

---

## 🚀 Installation & Running

### Install Python

PylaAI has been tested with:

```bash
Python 3.11.9
```

Download Python:

```
[Python 3.11](https://www.python.org/downloads/release/python-3119/)
```

---

###Run Universal Setup

run the smart installer:

```bash
python setup.py install
```

### Start Your Emulator

see how you can start your emulator in https://pyla-ai.pages.dev/#starting

---

### Launch PylaAI

Run the bot:

```bash
python main.py
```

---


### Localhost Mode

This open-source version runs in **localhost mode**.

The following cloud features are disabled by default:

- Login system
- Cloud statistics
- Auto updates
- Remote API services

---

### Running Tests

To make sure changes do not introduce regressions:

```bash
python -m unittest discover
```

---

## 📌 Project Links

- **[Discord](https://discord.gg/xUusk3fw4A)** Join the Pyla Server
- **{Trello](https://trello.com/b/SAz9J6AA/public-pyla-trello)**

---

## ⚖️ License

Please respect the **"No Selling" license** out of respect for the work of the official developers.

This project is **not permitted to be sold or monetized**.

---

## 👨‍💻 People that have been Official Developers

- **ivanyordanovgt**
- **AngelFireLA**
- **awarzu**

## Contributors

- **Maayan080**
- **simonrejzek**
- **bocchi-the-cat**
- **Ariko842**
