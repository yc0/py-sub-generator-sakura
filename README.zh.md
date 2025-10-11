# 🌸 Sakura 字幕生成器

一個功能強大、架構完善的應用程式，專為生成日語字幕並支援多語言翻譯而設計。採用現代 AI 模型和用戶友好的 Tkinter 介面構建。

## ✨ 功能特色

- **🎙️ 日語 ASR**：使用 OpenAI Whisper 進行高品質日語語音辨識
- **🌐 多語言翻譯**：使用 Hugging Face 模型翻譯成英語和繁體中文
- **🎯 雙語字幕輸出**：同時生成原始日語和翻譯字幕檔案，配備可自訂檔案名稱後綴
- **🖥️ 用戶友好的 GUI**：簡潔的 Tkinter 介面，方便視頻處理
- **⚙️ 可配置**：模型、設備和輸出偏好的全面設定
- **🏗️ 架構完善**：模組化設計，關注點清晰分離
- **📊 進度追蹤**：處理過程中的實時進度更新
- **💾 多種匯出格式**：SRT 字幕匯出，計劃支援更多格式
- **⚡ GPU 優先設計**：針對 NVIDIA CUDA 和 Apple Silicon MPS 加速最佳化

## 🎯 GPU 加速優先

本專案採用 **GPU 優先** 設計，針對硬體加速進行最佳化：

### **✅ 支援的 GPU 加速：**
- **🚀 NVIDIA CUDA**：RTX/GTX 顯卡的完整 GPU 加速
- **🍎 Apple Silicon MPS**：M1/M2/M3/M4 晶片的 Metal Performance Shaders
- **⚡ 自動偵測**：智能 GPU 選擇（CUDA > MPS > CPU）

### **⚠️ 純 CPU 使用者：**
如果您 **僅有 CPU** 並需要 CPU 最佳化推理，本專案專注於 GPU 加速。對於純 CPU 設置，建議考慮：

- **替代方案**：實作基於 `ctransformers` 的類別以進行 CPU 最佳化
- **當前重點**：本專案優先考慮 GPU 效能（MPS/CUDA）
- **效能**：GPU 提供比純 CPU 解決方案快 3-5 倍的推理速度
- **開發方向**：CPU 特定最佳化（ctransformers、GGUF 等）不是主要重點

### **建議硬體：**
- **Apple Silicon Mac**：M1/M2/M3/M4 配 16GB+ RAM
- **NVIDIA GPU**：RTX/GTX 顯卡配 8GB+ VRAM
- **最低要求**：16GB 系統 RAM 用於模型載入

## 🚀 硬體加速效能

**跨平台硬體加速**，顯著提升效能：

### **⚡ 音頻提取效能：**
| 平台 | 加速技術 | 效能 | 實時倍率 |
|----------|--------------|-------------|-----------------|
| Apple Silicon | VideoToolbox | **18.8 倍實時** | M1/M2/M3/M4 最佳化 |
| Windows/Linux | CUDA | **15-20 倍實時** | RTX/GTX GPU 加速 |
| 軟體回退 | CPU | **1.0 倍實時** | 通用相容性 |

### **🎯 ASR 處理效能：**
| 組件 | 效能提升 | 備註 |
|-----------|------------------|-------|
| ASR 處理 | 快 3-5 倍 | MPS/CUDA 加速 Whisper 模型 |
| 翻譯 | 快 2-4 倍 | 硬體最佳化 PyTorch 運算 |  
| 模型載入 | 快 50% | 使用 uv 最佳化依賴項 |
| 記憶體使用 | 減少 20-30% | 高效原生庫 |

### 硬體加速特色：
- **🏁 跨平台**：VideoToolbox（Apple）、CUDA（NVIDIA）、軟體回退
- **🔥 自動偵測**：智能硬體能力偵測
- **⚡ 音頻提取**：Apple Silicon 上高達 18.8 倍實時效能  
- **💚 通用相容性**：所有硬體 100% 回退支援
- **🧠 記憶體高效**：最佳化資源管理和清理

## 🚀 快速開始

### 硬體需求

#### **🎯 建議配置（GPU 加速）：**
- **Apple Silicon**：M2/M3/M4 配 16GB+ 統一記憶體
- **NVIDIA GPU**：RTX 3070/4070+ 配 8GB+ VRAM 
- **系統 RAM**：16GB+ 用於模型載入
- **儲存空間**：10GB+ 可用空間（3.6GB 模型 + 快取）
- **預期效能**：實時處理 8-12 倍

#### **💰 最低配置（經濟型設置）：**
- **Apple Silicon**：M1 配 8GB 統一記憶體
- **NVIDIA GPU**：GTX 1660 配 6GB VRAM
- **替代方案**：使用 `whisper-medium`（1.5GB vs 3GB）
- **預期效能**：實時處理 4-6 倍

#### **⚠️ 純 CPU（未最佳化）：**
- **效能**：比 GPU 慢 4-12 倍（實時處理 1-2 倍）
- **記憶體**：建議 32GB+ RAM 用於所有模型
- **模型**：考慮使用 `whisper-small` 以提升 CPU 效能
- **注意**：實作 ctransformers 進行 CPU 最佳化（超出專案範圍）

> 📝 **詳細規格請參見[預設模型與需求](#-預設模型與需求)章節**

### 軟體先決條件

- Python 3.8 或更高版本
- FFmpeg（用於音頻提取）
- GPU 驅動程式（NVIDIA 需要 CUDA 11.8+，Apple Silicon 需要最新 macOS）

### 安裝

#### 方法 1：純 uv（推薦 - 零環境污染）
```bash
git clone <repository-url>
cd py-sub-generator-sakura

# 無需安裝 - 在隔離環境中執行！
uv run python main.py
```
- 🔒 **零環境污染** - 完全隔離
- ⚡ **最快啟動** - 自動管理依賴項 
- 🛡️ **安全** - 不修改系統 Python
- 🧹 **乾淨** - 無安裝檔案殘留

#### 方法 2：通用設置
```bash
git clone <repository-url>
cd py-sub-generator-sakura

uv run python setup.py  # 使用 uv run 避免污染系統 Python！
# 或者（如果必須）：
python setup.py  # ⚠️ 如果 uv 不可用可能會安裝到系統 Python
```
- ✅ **適用於所有平台**（Windows、Linux、macOS）
- ✅ **自動偵測 uv** 以加快安裝
- ✅ **檢查系統需求**（Python、FFmpeg）
- ⚠️ **警告**：如果 uv 不可用可能會污染系統 Python

#### 方法 3：Apple Silicon 最佳化（M1/M2/M3 Mac）
```bash
git clone <repository-url>
cd py-sub-generator-sakura

python setup_apple_silicon.py      # 最大效能設置
```
- 🍎 **Apple Silicon 專屬最佳化**
- ⚡ **處理速度快 3-5 倍**，配備 MPS 加速
- 🔧 **自動 FFmpeg 安裝**，通過 Homebrew
- 💾 **記憶體使用減少 20-30%**

### 使用方式

#### GUI 模式（推薦）
```bash
# 使用 uv（最快，自動依賴管理）：
uv run python main.py

# 使用 setup.py 或 setup_apple_silicon.py 安裝後：
python main.py
```

#### CLI 模式（未來功能）
```bash
# 使用 uv（無需安裝）
uv run python main.py --no-gui video.mp4

# 安裝後使用
python main.py --no-gui video.mp4
```

## 🎯 雙語字幕功能

### **🌸 同時生成日語和翻譯字幕**

本應用程式支援**雙語字幕輸出**，完美適合語言學習和多語言內容創建：

#### ✨ **功能特色：**
- **🎌 日語原始字幕**：保留原始 ASR 轉錄結果
- **🌐 翻譯字幕**：同時生成目標語言（如中文、英文）字幕
- **⚙️ UI 控制**：設定對話框中的便捷切換選項
- **📝 自訂檔案名稱**：可配置的檔案後綴（如：`_ja.srt`、`_zh.srt`）
- **🎯 彈性輸出**：根據需要選擇生成單一或雙語字幕

#### 🛠️ **使用方式：**

**透過 GUI 設定：**
1. 點擊 **⚙️ 設定** 按鈕
2. 在設定對話框中找到 **"雙語輸出"** 選項
3. 勾選 **"生成兩種語言的字幕"**
4. 可選：點擊 **"進階選項"** 自訂檔案後綴
5. 設定會自動儲存

**檔案輸出範例：**
```
video.mp4 → 
├── video_ja.srt    (日語原始字幕)
└── video_zh.srt    (中文翻譯字幕)
```

**進階自訂：**
- **日語後綴**：預設 `_ja`（可改為 `_japanese`、`_orig` 等）
- **翻譯後綴**：預設 `_zh`（可改為 `_chinese`、`_trans` 等）
- **檔案命名**：完全可自訂的工作流程整合

#### 🎬 **完美適用場景：**
- **🎓 語言學習**：對照原文和翻譯學習
- **📺 多語言內容**：為不同觀眾提供選擇
- **🔄 翻譯比對**：檢查翻譯品質和準確性
- **📚 字幕製作**：專業字幕製作工作流程

## 🏗️ 架構

應用程式採用乾淨的模組化架構：

```
src/
├── models/          # 數據模型和結構
├── utils/           # 工具程式（設定、日誌、檔案處理、音頻處理）  
├── asr/             # 自動語音辨識模組
├── translation/     # 翻譯管線和模型
├── subtitle/        # 字幕處理和生成
└── ui/              # Tkinter GUI 組件
```

### 核心組件

#### 🎯 **字幕生成器** (`src/subtitle/subtitle_generator.py`)
協調整個字幕生成管線的主要編排器：
- 視頻驗證和元資料提取
- 音頻提取和預處理  
- 長視頻的分塊 ASR 轉錄
- 多語言翻譯
- 字幕格式化和匯出

#### 🎤 **ASR 模組** (`src/asr/`)
- **基礎 ASR** (`base_asr.py`)：ASR 實作的抽象介面
- **Whisper ASR** (`whisper_asr.py`)：OpenAI Whisper 整合，支援批次處理

#### 🌐 **翻譯模組** (`src/translation/`)
- **翻譯管線** (`translation_pipeline.py`)：協調多階段翻譯
- **HuggingFace 翻譯器** (`huggingface_translator.py`)：基於 Transformer 的 GPU 翻譯
- **PyTorch 翻譯器** (`pytorch_translator.py`)：GPU 優先的 PyTorch 實作
- **多階段翻譯器**：日語 → 英語 → 繁體中文

##### 翻譯後端策略：
- **🚀 主要**：帶 GPU 加速的 HuggingFace Transformers（MPS/CUDA） 
- **🌸 進階**：SakuraLLM 整合的 PyTorchTranslator（專業日語 LLM）
- **⚡ 替代**：針對任何 HuggingFace 模型的純 PyTorch GPU 最佳化
- **❌ 不包含**：ctransformers（僅 CPU 重點與專案目標衝突）
- **🎯 重點**：實時字幕生成的最大 GPU 效能

## ⚙️ 配置

應用程式使用具有合理預設值的 JSON 配置檔案：

```json
{
  "asr": {
    "model_name": "openai/whisper-large-v3",
    "device": "auto",
    "chunk_length": 30
  },
  "translation": {
    "ja_to_en_model": "Helsinki-NLP/opus-mt-ja-en", 
    "en_to_zh_model": "Helsinki-NLP/opus-mt-en-zh",
    "batch_size": 8
  },
  "dual_language": {
    "generate_both_languages": false,
    "japanese_suffix": "_ja",
    "translated_suffix": "_zh"
  },
  "ui": {
    "window_size": "800x600",
    "theme": "default"
  }
}
```

配置可透過以下方式修改：
- GUI 設定對話框（⚙️ 設定按鈕）
- 直接編輯 JSON 檔案（`config.json`）
- 命令列參數

## 🤖 預設模型與需求

### 🎙️ **ASR 模型（語音辨識）**

#### **預設：OpenAI Whisper Large-v3**
- **模型**：[`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)
- **大小**：~3GB 下載
- **語言**：99+ 種語言（針對日語最佳化）
- **品質**：日語語音辨識的最高準確度

**硬體需求：**
| 硬體 | **最低** | **建議** |
|----------|-------------|----------------|
| **Apple Silicon** | M1 8GB | M2/M3 16GB+ |
| **NVIDIA GPU** | GTX 1660 6GB | RTX 3070 8GB+ |
| **純 CPU** | 16GB RAM | 32GB RAM |
| **處理速度** | 實時 2 倍 | 實時 8-12 倍 |

### 🌐 **翻譯模型**

#### **日語 → 英語**
- **模型**：[`Helsinki-NLP/opus-mt-ja-en`](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)
- **大小**：~300MB 下載
- **專長**：日語到英語翻譯
- **效能**：針對字幕長度文字最佳化

#### **英語 → 中文（繁體）**
- **模型**：[`Helsinki-NLP/opus-mt-en-zh`](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh)
- **大小**：~300MB 下載  
- **專長**：英語到中文翻譯
- **輸出**：繁體中文字符

## 📋 工作流程

1. **🎬 視頻輸入**：透過檔案瀏覽器選擇視頻檔案
2. **⚙️ 配置**：選擇目標語言和設定
3. **🔍 驗證**：驗證視頻格式並提取元資料  
4. **🎵 音頻提取**：使用 FFmpeg 提取音頻
5. **🎙️ ASR 處理**：使用 Whisper 生成日語轉錄
6. **🌐 翻譯**：翻譯成英語和繁體中文
7. **📝 字幕生成**：建立格式化的字幕檔案
8. **👁️ 預覽與匯出**：檢查結果並匯出 SRT 檔案

## 🎛️ 進階功能

### 分塊處理
- 自動將長視頻分割成塊以提高記憶體效率
- 重疊塊防止詞邊界問題
- 可配置的塊大小和重疊

### 設備管理  
- 自動 GPU 偵測並回退到 CPU
- 記憶體感知的模型載入/卸載
- 可配置的設備偏好

### 字幕最佳化
- 自動文字清理和格式化
- 最佳斷行以提高可讀性  
- 可配置的時間約束

## 📊 效能考慮

- **記憶體使用**：根據需要載入/卸載模型
- **GPU 支援**：可用時啟用 CUDA 加速
- **批次處理**：翻譯的高效批次處理
- **分塊**：處理任何長度的視頻

## 🔮 未來增強功能

- [ ] CLI 模式實作
- [ ] 額外字幕格式（VTT、ASS）
- [ ] 說話者分離
- [ ] 自訂模型微調
- [ ] Web 介面
- [ ] Docker 部署
- [ ] 額外語言對

## 📚 文檔

### 📋 專案文檔
- **[進度摘要](PROGRESS_SUMMARY.md)** - 最近改進和轉換的完整概述
- **[技術更新日誌](TECHNICAL_CHANGELOG.md)** - 詳細的技術實作文檔  
- **[UV 工具策略](docs/UV_TOOL_STRATEGY.md)** - 為什麼我們使用 `uv tool` 而不是專案依賴項進行開發工具

### 🚀 最近的重大更新
- **原生 Whisper 實作** - 從管線重構到原生 `generate()` 方法
- **零實驗警告** - 消除所有 ASR 警告以實現乾淨操作  
- **生產就緒** - 遵循 Whisper 論文最佳實務（第 3.8 節）
- **效能最佳化** - 更好的記憶體使用和生成效率
- **雙語字幕生成** - 支援同時輸出日語和翻譯字幕檔案

### 🎯 關鍵改進
- ✅ **無令牌限制** - 移除人為的生成約束
- ✅ **靜默操作** - 乾淨的日誌，無實驗警告
- ✅ **更好的品質** - 使用 Whisper 的預期架構
- ✅ **完整設備支援** - 維持 MPS/CUDA/CPU 相容性
- ✅ **雙語輸出** - 可配置的日語+翻譯字幕檔案生成

*完整技術詳情請參見 [TECHNICAL_CHANGELOG.md](TECHNICAL_CHANGELOG.md)*

## 🤝 貢獻

程式碼庫設計具有可擴展性：

- **新 ASR 模型**：擴展 `BaseASR` 類別
- **新翻譯模型**：擴展 `BaseTranslator` 類別  
- **新匯出格式**：在 `SubtitleFile` 類別中新增方法
- **新 UI 組件**：新增到 `src/ui/components/`

## 📄 授權

[授權資訊]

## 🙏 致謝

- **OpenAI** 提供 Whisper ASR 模型和基礎研究
- **Whisper 論文（第 3.8 節）** - "Robust Speech Recognition via Large-Scale Weak Supervision" 提供實作方法論和最佳實務
- **SakuraLLM 專案** 提供針對輕小說和動漫內容最佳化的專業日中翻譯模型
- **Hugging Face** 提供 transformer 模型和管線 API
- **Helsinki-NLP** 提供 OPUS-MT 翻譯模型
- **FFmpeg 專案** 提供全面的音頻處理能力

---

**用 ❤️ 為字幕生成社群而建** 🌸