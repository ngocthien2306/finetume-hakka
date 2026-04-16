# Dataset Analysis — Hakka Translation Fine-tuning

> Cập nhật: 2026-04-16
> Dự án: `finetume-hakka` — Fine-tune Llama3-Taiwan-8B cho bài toán dịch Hakka ↔ Mandarin

---

## Tổng quan

| # | Dataset | Loại dữ liệu | Số lượng | Dialect | Có Mandarin? | Dialogue? |
|---|---------|--------------|----------|---------|--------------|-----------|
| 1 | `HAKKA_UTF8_TSV/` | **Multi-turn dialogue** (medical) | 294 turns / **143 rounds** | 四縣 | ✅ | ✅ |
| 2 | `dict-hakka.json` | Sentence (general) | ~20,185 pairs | Đa dialect | ✅ | ❌ |
| 3 | `HAT-Lexicon-Sixian.csv` | Word-level | ~53,362 pairs | 四縣 | ✅ | ❌ |
| 3 | `HAT-Lexicon-Hailu.csv` | Word-level | ~50,690 pairs | 海陸 | ✅ | ❌ |
| 4 | `dataset/json/` (**MỚI**) | Sentence (speech transcripts) | **22,841 unique** (103,533 files) | 100% 海陸 | ✅ | ❌ |
| 5 | `客語漢字QA/` (**MỚI**) | **Multi-turn dialogue** (Hakka-only) | 888 turns / **434 rounds** | 四縣 | ❌ | ✅ |

---

## Source 1: HAKKA_UTF8_TSV

### Mô tả
Bộ hội thoại y tế song ngữ **Hakka ↔ Mandarin**, định dạng multi-turn dialogue (Doctor ↔ Patient). Mỗi lượt hội thoại có đầy đủ: Hakka漢字, Hakka拼音, và Mandarin.

> **Encoding thực tế**: UTF-16-LE (không phải UTF-8 như tên gọi) — cần `open(f, encoding='utf-16-le')`.

### Cấu trúc thư mục
```
dataset/HAKKA_UTF8_TSV/
├── 腎臟泌尿系統1.txt   ← Thận / tiết niệu (3 files)
├── 腎臟泌尿系統2.txt
├── 腎臟泌尿系統3.txt
├── 軀幹四肢關節1.txt   ← Cơ thể / khớp (9 files)
├── 軀幹四肢關節2.txt
...
└── 軀幹四肢關節9.txt   (12 files tổng)
```

### Định dạng (6 cột, tab-separated)
```
音訊檔案名稱          發音人      腔調  客語拼音          客語漢字                  華語翻譯
B-1_speaker01_001.wav  speaker01  四縣  iˊ siiˊ：nˇ hoˋ  你好，請問哪位毋鬆爽啊？  您好，請問哪裡不舒服啊？
B-1_speaker02_001.wav  speaker02  四縣  piang nginˇ：...  先生，𠊎這駁仔密密愛屙尿  醫生，我最近頻尿得很厲害...
```

| Cột | Mô tả |
|-----|-------|
| `音訊檔案名稱` | Tên file audio |
| `發音人` | `speaker01` = Doctor, `speaker02` = Patient |
| `腔調` | Dialect (100% 四縣) |
| `客語拼音` | Hakka romanization |
| `客語漢字` | Hakka Chinese characters |
| `華語翻譯` | Mandarin translation |

### Thống kê
| Mục | Giá trị |
|-----|---------|
| Tổng files | **12** `.txt` |
| Tổng turns (utterances) | **294** |
| Tổng rounds (D+P pairs) | **143** |
| Translation pairs | **294** (mỗi turn có Hakka + Mandarin) |
| Doctor turns (speaker01) | 141 |
| Patient turns (speaker02) | 153 |
| Dialect | 100% **四縣 (Sixian)** |
| Domain | Y tế (thận/tiết niệu + cơ xương khớp) |

### Rounds per file
| File | Rounds | File | Rounds |
|------|--------|------|--------|
| 腎臟泌尿系統1 | 7 | 軀幹四肢關節1 | 22 |
| 腎臟泌尿系統2 | 8 | 軀幹四肢關節2 | 19 |
| 腎臟泌尿系統3 | 7 | 軀幹四肢關節3 | 13 |
| | | 軀幹四肢關節4-9 | 10–12 |

### Quan hệ với 客語漢字QA (Source 5)
**Phát hiện quan trọng**: HAKKA_UTF8_TSV và 客語漢字QA chứa **cùng 12 dialogues** (round count khớp 100%). Điểm khác biệt duy nhất:

| | HAKKA_UTF8_TSV | 客語漢字QA |
|-|----------------|-----------|
| Dialogues chung | 腎臟(3) + 軀幹(9) = **12** | 腎臟(3) + 軀幹(9) = **12** |
| Thêm topics | ❌ | 全身性症狀(10) + 腹部腸胃肝膽(13) = **+23** |
| Có Mandarin | ✅ **Có** | ❌ Không |
| Có Hakka拼音 | ✅ **Có** | ❌ Không |

> `客語漢字QA` = Hakka-only superset; `HAKKA_UTF8_TSV` = subset có Mandarin + phiên âm.

### Ưu / Nhược điểm
- **Ưu**: Dataset duy nhất vừa là **dialogue** vừa có **Mandarin translation** — dùng được cho cả 2 task
- **Nhược**: Nhỏ (294 turns / 143 rounds), chỉ 2 topic, chỉ 四縣; encoding UTF-16-LE dễ gây lỗi

---

## Source 2: dict-hakka.json

### Mô tả
Từ điển Hakka chính thức (Bộ Giáo dục Đài Loan), chứa ~14,712 mục từ với ví dụ câu song ngữ.

### Cấu trúc JSON
```json
{
  "title": "詞條 (từ đầu mục)",
  "heteronyms": [
    {
      "audio_id": "03773",
      "pinyin": "四⃞gia²⁴ 海⃞gia⁵³ ...",
      "definitions": [
        {
          "example": [
            "￹Hakka sentence。￻Mandarin translation。"
          ],
          "def": "Định nghĩa...",
          "type": ""
        }
      ]
    }
  ]
}
```

**Annotation markers:**
- `\ufff9` (`￹`) = bắt đầu Hakka
- `\ufffb` (`￻`) = phân cách Hakka / Mandarin (Mandarin theo sau)

### Thống kê
| Mục | Giá trị |
|-----|---------|
| Tổng từ điển entries | 14,712 |
| Tổng example sentences | **20,185** |
| Có annotation Hakka+Mandarin | 20,185 (100%) |
| Dialect | Đa dialect (四縣, 海陸, 平原, 安西) — ghi chú trong pinyin |
| Domain | Đa dạng (general language) |

### Ưu / Nhược điểm
- **Ưu**: Nguồn chính thức, coverage rộng, ví dụ tự nhiên, context phong phú
- **Nhược**: Cần parse Unicode annotation markers; một số ví dụ rất ngắn (thiếu Mandarin)

---

## Source 3: HAT Lexicon (Sixian + Hailu)

### Mô tả
Bộ từ điển HAT (Hakka Annotation Tool) mức word-level, cover 2 dialect chính.

### Cấu trúc CSV
```
華語詞,序號,客語漢字,客語拼音,標籤,備註
一些,0,一息仔,id2 xid5 e31,,
一切,0,一切,id2 qied2,,
```

| Cột | Mô tả |
|-----|-------|
| `華語詞` | Từ Mandarin |
| `客語漢字` | Từ Hakka (chữ Hán) |
| `客語拼音` | Phiên âm Hakka |
| `標籤` | Tag (thường trống) |
| `備註` | Ghi chú |

### Thống kê
| File | Dialect | Rows | Unique pairs |
|------|---------|------|--------------|
| `HAT-Lexicon-Sixian.csv` | 四縣 (Sixian) | 53,363 | ~53,362 |
| `HAT-Lexicon-Hailu.csv` | 海陸 (Hailu) | 50,691 | ~50,690 |
| **Tổng** | 四縣 + 海陸 | **~104,053** | ~104,052 |

### Ưu / Nhược điểm
- **Ưu**: Rất lớn (~104k entries), cover cả 2 dialect chính
- **Nhược**: Chỉ là **word/phrase level** — không phải câu hoàn chỉnh; ít context; không phù hợp làm val set đánh giá dịch câu

---

## Source 4: dataset/json/ (MỚI — Chưa dùng trong notebook)

### Mô tả
Metadata của bộ ngữ liệu ghi âm tiếng Hakka quy mô lớn từ Đài Loan. Mỗi file JSON tương ứng 1 segment ghi âm, chứa transcript song ngữ (Hakka ↔ Mandarin) cùng thông tin speaker.

### Cấu trúc thư mục
```
dataset/json/
├── HF3010001E1009/          ← HF = Female speaker, 301 = speaker ID, E1009 = card type
│   ├── HF3010001E1009_1.json
│   ├── HF3010001E1009_2.json
│   └── ...
├── HM3010001E1009/          ← HM = Male speaker
│   └── ...
└── ... (657 thư mục tổng)
```

**Naming convention:**
- `HF` / `HM` = Female / Male speaker
- `3XX` = Speaker serial number
- `0001` = Session number
- `E/F/G/H` = Card type (kịch bản prompt)
- `XXXX` = Card ID number

### Cấu trúc JSON
```json
{
  "id": 145260,
  "客語漢字": "頭前斡轉正片就到新竹市政府咧！",
  "客語拼音": "teu55 cien55 vad5 zhon24 zhin11 pien24 ciu33 do11 sin53 zhug5 shi11 zhin11 fu24 le53 ！",
  "華語字":   "前面右轉就到新竹市政府了！",
  "錄音腔調": "海陸",
  "流暢度":   "普通",
  "提示卡編號": "E1009",
  "發音員編號": "HF301",
  "性別":     "女性",
  "年齡":     60,
  "教育程度": "大學",
  "18歲前居住地": "桃園市中壢區",
  "現居地":   "桃園市新屋區",
  "身分別":   "薪傳師（教師）",
  "錄音環境": "一般辦公室",
  "音檔編號": 1,
  "音檔長度": "8.59",
  "內容類型": "錄音稿"
}
```

### Thống kê tổng thể
| Mục | Giá trị |
|-----|---------|
| Tổng JSON files | **103,533** |
| Unique (客語漢字, 華語字) pairs | **22,841** |
| Dialect | **100% 海陸 (Hailu)** |
| Có cả Hakka漢字 + Hakka拼音 + Mandarin | **100%** |
| Speaker Female (HF) | 61,819 files |
| Speaker Male (HM) | 41,714 files |
| Thư mục speaker | 657 |

### Phân phối Card Type (kịch bản)
| Card Type | Files | Mô tả suy đoán |
|-----------|-------|----------------|
| **F** | 45,376 (43.8%) | Conversational / dialogue prompts |
| **G** | 33,916 (32.8%) | Situational / daily life |
| **E** | 13,590 (13.1%) | Instruction / directional |
| **H** | 10,651 (10.3%) | Narrative / story |

### Phân phối Fluency (流暢度)
| Fluency | Files | % | Khuyến nghị dùng cho training |
|---------|-------|---|-------------------------------|
| 流暢 (Fluent) | 17,582 | 17% | ✅ Ưu tiên |
| 普通 (Normal) | 84,052 | 81% | ✅ Dùng được |
| 不流暢 (Disfluent) | 1,899 | 2% | ⚠️ Nên loại bỏ |

### Ưu / Nhược điểm
- **Ưu**:
  - Quy mô lớn nhất trong tất cả sources (~22k unique sentence pairs)
  - Câu hoàn chỉnh, ngữ cảnh tự nhiên từ nhiều domain (daily life, directions, dialogue...)
  - Có **cả 客語漢字 lẫn 客語拼音** → train được 2 input format
  - Metadata phong phú: gender, age, fluency, location → có thể làm subgroup evaluation
  - Nhiều speaker đọc cùng câu → data augmentation tự nhiên
- **Nhược**:
  - Toàn bộ chỉ **海陸 dialect** — không có 四縣
  - Cần dedup: nhiều speaker đọc cùng prompt card → cùng câu xuất hiện nhiều lần
  - Raw files phân tán trong 657 thư mục

---

---

## Source 5: 客語漢字QA/ (MỚI — Chưa dùng trong notebook)

### Mô tả
Bộ hội thoại y tế **chỉ có tiếng Hakka** (không có bản dịch Mandarin). Dữ liệu là các kịch bản khám bệnh doctor-patient, luân phiên theo lượt.

> **Xác nhận từ tác giả**: 4 topic, mỗi `.tsv` ≤ 12 rounds. Row 1 = Language Name (header), Row 2 = Doctor, Row 3 = Patient, lặp đến hết.
> **Mục đích đề xuất**: Train **Hakka medical dialogue** trên model `yling-Taiwan-8B` bằng sliding window hoặc full conversation.

### Cấu trúc thư mục
```
dataset/客語漢字QA/
├── 全身性症狀1.tsv      ← Triệu chứng toàn thân (10 files)
├── 腎臟泌尿系統1.tsv    ← Thận / tiết niệu (9 files)
├── 腹部腸胃肝膽1.tsv    ← Bụng / tiêu hoá / gan mật (13 files)
└── 軀幹四肢關節1.tsv    ← Cơ thể / khớp (9 files)
```

### Định dạng file (xác nhận từ tác giả)
```
客語漢字                                    ← Row 1: Language Name (header)
恁會早，請坐。今晡日仰仔呢？                   ← Row 2: Doctor
先生，𠊎對昨暗晡開始作燒...                   ← Row 3: Patient
恁樣哦。除忒作燒以外，還有其他个症頭無？         ← Row 4: Doctor
喉嗹頭有一息仔痛，還過歸身仔痠痛...             ← Row 5: Patient
...                                         ← Repeat Doctor/Patient
```

- **1 cột duy nhất**: `客語漢字`
- **Không có Mandarin**, không có phiên âm
- 1 round = 1 Doctor turn + 1 Patient turn

### Thống kê
| Mục | Giá trị |
|-----|---------|
| Tổng files (dialogues) | 41 |
| Tổng utterances | 888 |
| Tổng rounds (D+P pairs) | **434** |
| Rounds per dialogue | min=7, max=22, **avg=10.6** |
| Files ≤ 12 rounds | 38/41 (93%) |
| Files > 12 rounds (軀幹四肢關節1-3) | 3/41 |
| Domain | Y tế (4 topics) |
| Dialect | 四縣 (suy đoán, cùng hệ với HAKKA_UTF8_TSV) |

### Phân phối theo topic
| Topic | Files | Utterances | Rounds |
|-------|-------|------------|--------|
| 全身性症狀 (Triệu chứng toàn thân) | 10 | 211 | 105 |
| 腹部腸胃肝膽 (Bụng / tiêu hoá / gan mật) | 13 | 273 | 136 |
| 軀幹四肢關節 (Cơ thể / khớp) | 9 | 247 | 123 |
| 腎臟泌尿系統 (Thận / tiết niệu) | 9 | 157 | 78 |
| **Tổng** | **41** | **888** | **434** |

### Chiến lược training (theo gợi ý của tác giả)

#### Option A: Sliding Window
Tạo nhiều examples ngắn từ mỗi dialogue bằng cách trượt cửa sổ.

```
Dialogue: [D1, P1, D2, P2, D3, P3, ...]

Window=3 (input 3 turns → predict turn 4):
  [D1, P1, D2] → predict P2
  [P1, D2, P2] → predict D3
  [D2, P2, D3] → predict P3
  ...

Window=6 (input 6 turns → predict turn 7):
  [D1,P1,D2,P2,D3,P3] → predict D4
  ...
```

| Strategy | Training examples | Ghi chú |
|----------|-------------------|---------|
| Sliding window = 3 turns | **765 examples** | Nhiều nhất, context ngắn |
| Sliding window = 6 turns | **642 examples** | Balance context/data |
| Full conversation (≤12 rounds) | **41 examples** | Ít nhất, context đầy đủ |

#### Option B: Full Conversation (12 rounds)
Mỗi file là 1 training example hoàn chỉnh — model học toàn bộ luồng khám bệnh.

```
System: Bạn là bác sĩ nói tiếng Hakka (四縣腔)...
User:   [Patient turn 1]
Assistant: [Doctor turn 2]
User:   [Patient turn 3]
...
```

### Quan hệ với HAKKA_UTF8_TSV (Source 1)
| Mục | HAKKA_UTF8_TSV | 客語漢字QA |
|-----|----------------|-----------|
| Topics | 腎臟 + 軀幹 (2 topics) | 4 topics |
| Có Mandarin | ✅ Có | ❌ Không |
| Encoding | Non-UTF8 | UTF-8 |
| Mục đích | Translation pairs | Dialogue generation |

### Ưu / Nhược điểm
- **Ưu**:
  - Dialogue tự nhiên, turn-taking rõ ràng (Doctor ↔ Patient)
  - Cover 4 topic y tế, trung bình ~10 rounds/dialogue
  - Phù hợp train **Hakka conversational AI** cho domain y tế
  - Sliding window tạo được 642–765 training examples từ 41 dialogues
- **Nhược**:
  - **Không có Mandarin** → không dùng trực tiếp cho translation
  - Tổng thể nhỏ (888 utterances)
  - Không có phiên âm

### Khả năng sử dụng
| Use case | Khả thi? | Examples |
|----------|----------|---------|
| Translation training (H↔M) | ❌ Không trực tiếp | — |
| Hakka dialogue generation | ✅ Sliding window | 642–765 |
| Full medical conversation | ✅ Full conv | 41 |
| Synthetic pairs (via translation model) | ✅ Có thể | ~434 new pairs |

---

## So sánh tổng hợp

```
Sentence-level pairs (Hakka ↔ Mandarin, dùng cho translation):
┌──────────────────────────────────────────────────────────────────┐
│ Source 1: HAKKA_UTF8_TSV       █░░░░░░░░░░░░░░░░░  ~294 pairs   │
│ Source 2: dict-hakka.json      ████████████████░░  ~20,185      │
│ Source 4: JSON (new, unique)   ████████████████████ ~22,841      │
└──────────────────────────────────────────────────────────────────┘

Word-level pairs (bổ trợ vocab):
┌──────────────────────────────────────────────────────────────────┐
│ Source 3: HAT Sixian           ██████████  ~53,362 word pairs    │
│ Source 3: HAT Hailu            █████████   ~50,690 word pairs    │
└──────────────────────────────────────────────────────────────────┘

Hakka-only dialogue (không dùng trực tiếp cho translation):
┌──────────────────────────────────────────────────────────────────┐
│ Source 5: 客語漢字QA            █░░░░░░░░░  888 utterances        │
└──────────────────────────────────────────────────────────────────┘
```

### Dialect coverage
| Dialect | Source | Sentence pairs |
|---------|--------|----------------|
| 四縣 (Sixian) | TSV + dict + HAT-Sixian | ~20k+ sentences + ~53k words |
| 海陸 (Hailu) | JSON + HAT-Hailu | **~22k sentences** + ~51k words |
| Cả hai | Notebook hiện tại | Chủ yếu 四縣 |

> **Điểm quan trọng**: Dataset JSON mới **bổ sung lớn cho 海陸 dialect** — phần yếu nhất trong notebook hiện tại.

---

## Khuyến nghị tích hợp Source 4

### Strategy 1: Filter quality
```python
# Chỉ dùng fluent sentences
df = df[df['流暢度'] != '不流暢']  # loại bỏ 1,899 files (~2%)
```

### Strategy 2: Dedup
```python
# Dedup theo nội dung (nhiều speaker đọc cùng câu)
df = df.drop_duplicates(subset=['客語漢字', '華語字'])
# → còn ~22,841 unique pairs
```

### Strategy 3: Dual input format
```python
# Format A: Dịch từ Hakka chữ Hán
"Translate Hakka (Hailu) to Mandarin: 頭前斡轉就到新竹市政府咧！"
→ "前面右轉就到新竹市政府了！"

# Format B: Dịch từ Hakka phiên âm
"Translate Hakka romanization to Mandarin: teu55 cien55 vad5 zhon24..."
→ "前面右轉就到新竹市政府了！"
```

### Strategy 4: Split gợi ý
| Split | Criteria | Size |
|-------|----------|------|
| Train | 流暢 + 普通, all card types | ~21,000 |
| Val | 流暢 only, random sample | ~800 |
| Test | 流暢 only, reserved | ~1,000 |

---

## Tổng kết

| Dataset | Notebook? | Train dịch? | Ghi chú |
|---------|-----------|-------------|---------|
| `HAKKA_UTF8_TSV` | ✅ Có | ✅ Tốt | Nhỏ (~294), 四縣, y tế |
| `dict-hakka.json` | ✅ Có | ✅ Tốt | 20k pairs, cần parse annotation |
| `HAT Lexicon ×2` | ✅ Có | ⚠️ Word-level | 104k từ, bổ trợ vocab |
| `dataset/json/` | ❌ Chưa | ✅ **Rất tốt** | **22k unique pairs, 海陸, có phiên âm** |
| `客語漢字QA/` | ❌ Chưa | ❌ Không trực tiếp | 888 utterances, **Hakka-only**, không có Mandarin |

**Kết luận**:
- **Dialogue generation** (có Mandarin): chỉ có `HAKKA_UTF8_TSV` — 143 rounds, 四縣, bilingual (Hakka + Mandarin mỗi turn)
- **Dialogue generation** (Hakka-only): `客語漢字QA` — 434 rounds, lớn hơn 3× nhưng thiếu Mandarin
- **Translation training**: ưu tiên tích hợp `dataset/json/` (22k unique pairs, 海陸)
- **Điểm đặc biệt**: `HAKKA_UTF8_TSV` ⊂ `客語漢字QA` — 12 dialogues chung, nhưng TSV có thêm Mandarin + phiên âm
