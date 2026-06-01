# PDF → Markdown 変換 & RAG CLIツール群

doclingを用いてPDFをMarkdownに変換し、HuggingFace埋め込みモデルでFAISSベクトルインデックスを構築・検索するCLIツール群。

**詳細ドキュメント**:
- [詳細利用マニュアル](docs/usage-manual.md) — CLIとGUIの操作手順・使用例
- [プログラムフロー資料](docs/program-flow.md) — 各プログラムの処理フロー・データ構造

## 必要環境

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- 環境変数 `OPENAI_API_KEY`（`rag_cli.py` の実行、および `convert.py` の画像説明文生成に必要）

## セットアップ

```powershell
uv sync
```

---

## convert.py — PDF → Markdown 変換

```
uv run python convert.py INPUT_PDF OUTPUT_DIR [OPTIONS]
```

### 基本変換

```powershell
uv run python convert.py report.pdf output/
```

成功すると標準出力に生成されたMarkdownのパスが表示される:

```
C:\path\to\output\report\report.md
```

出力構成（入力PDFと同名のサブフォルダが自動作成される）:

```
output/
└── report/
    └── report.md
```

### 画像を抽出して埋め込む

```powershell
uv run python convert.py report.pdf output/ --extract-images
```

出力構成:

```
output/
└── report/
    ├── report.md
    └── images/
        ├── picture-1.png
        └── picture-2.png
```

### 画像説明文をLLMで生成する

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run python convert.py report.pdf output/ --extract-images --describe-images
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--extract-images` | `False` | 図・数式をPNG画像として保存し、Markdownから相対パス参照する |
| `--describe-images` | `False` | LLMによる画像説明文をMarkdownに追記する（`--extract-images` 必須） |
| `--model MODEL_NAME` | `gpt-5-mini` | 使用するOpenAIモデル名 |
| `--image-dir DIR` | `images` | 画像保存先サブディレクトリ名 |
| `--verbose` | `False` | 詳細ログをSTDERRに出力する |

### 終了コード

| コード | 意味 |
|---|---|
| `0` | 成功 |
| `1` | 入力ファイルエラー（ファイル未存在など） |
| `2` | docling変換エラー |
| `3` | APIキーエラー（`OPENAI_API_KEY` 未設定） |

---

## build_faiss.py — MarkdownからFAISSインデックス構築

`convert.py` が出力したMarkdownファイルをベクトル化し、FAISSインデックスとして保存する。後続のRAGから `FAISS.load_local()` で再ロード可能。

```
uv run python build_faiss.py [OPTIONS]
```

### フォルダ全体をインデックス化

```powershell
uv run python build_faiss.py --markdown-dir ./docs
```

`./docs/faiss_index/` にインデックスが作成される:

```
docs/
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

### 特定ファイルをインデックス化

```powershell
uv run python build_faiss.py --markdown-file docs/chapter1.md docs/chapter2.md
```

インデックスの保存先は、指定ファイルの親フォルダが1つの場合はその親フォルダ内（例: `docs/faiss_index/`）、複数の親フォルダが混在する場合はカレントディレクトリ（`./faiss_index/`）に作成される。

### 既存インデックスを上書き

```powershell
uv run python build_faiss.py --markdown-dir ./docs --faiss-existing overwrite
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--markdown-dir PATH` | — | Markdownフォルダ（全`.md`ファイルをサブフォルダも含め再帰的に対象） |
| `--markdown-file FILE...` | `[]` | 特定Markdownファイル（複数指定可） |
| `--embedding-model NAME` | `cl-nagoya/ruri-v3-30m` | HuggingFace埋め込みモデル名 |
| `--chunk-size N` | `1000` | テキスト分割サイズ（文字数） |
| `--chunk-overlap N` | `200` | チャンクオーバーラップ（文字数） |
| `--faiss-existing MODE` | `skip` | 既存インデックスの扱い（`skip` / `overwrite`） |
| `--output-subdir NAME` | `faiss_index` | 保存先サブフォルダ名 |

**必須**: `--markdown-dir` または `--markdown-file` のいずれか一方または両方を指定すること。

### 終了コード

| コード | 意味 |
|---|---|
| `0` | 成功（インデックス構築完了、またはskipで既存インデックスあり） |
| `1` | エラー（引数エラー・ファイル未存在・処理中断等） |

### インデックスの利用（後続RAG）

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-v3-30m")
vectorstore = FAISS.load_local(
    "./docs/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
results = vectorstore.similarity_search("検索クエリ", k=3)
```

---

## rag_cli.py — FAISSインデックス検索・RAG回答生成

`build_faiss.py` が構築したFAISSインデックスを検索し、OpenAI LLMが回答と参照情報を生成するCLIツール。

```
uv run python rag_cli.py [OPTIONS] QUERY
```

### 基本的な使用方法

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run python rag_cli.py "RAGとは何ですか？"
```

出力例:

```
[回答]
RAG（Retrieval-Augmented Generation）とは、外部の知識ベースから関連情報を検索し、
その情報をもとに言語モデルが回答を生成する手法です。

[参照情報]
source: C:/path/to/docs/overview.md, chunk_id: 2
source: C:/path/to/docs/chapter1.md, chunk_id: 0
```

### 探索ルートフォルダを指定

```powershell
uv run python rag_cli.py --base-dir C:\Users\user\projects "RAGとは何ですか？"
```

### 複数インデックスフォルダを横断検索

```powershell
uv run python rag_cli.py `
  --base-dir C:\Users\user\projects `
  --index-folder faiss_index docs_index `
  "設計の概要を教えてください"
```

### パラメータを指定

```powershell
uv run python rag_cli.py --model gpt-4o --top-k 3 "最新の変更点は？"
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `QUERY` | — | 質問テキスト（必須） |
| `--base-dir PATH` | カレントディレクトリ | インデックス探索を開始するルートフォルダ |
| `--index-folder NAME...` | `faiss_index` | 探索するインデックスフォルダ名（複数指定可） |
| `--model NAME` | `gpt-5-mini` | 使用するAIモデル名 |
| `--top-k N` | `5` | 取得する参照文書数（1以上） |
| `--embedding-model NAME` | `cl-nagoya/ruri-v3-30m` | 埋め込みモデル名（インデックス構築時と同一にすること） |
| `--verbose` | `False` | 警告メッセージを含む詳細ログを表示する |

### インデックスが見つからない場合の動作

`--base-dir` 以下にFAISSインデックスが1件も見つからなかった場合、エラー終了はせず以下の警告を表示してからRAGなし（コンテキストなし）でLLMに問い合わせる。

```
警告: 有効なインデックスが見つかりませんでした。インデックスなしでAIに問い合わせます。
注意: ドキュメントに基づかない回答が生成される可能性があります。
```

### 終了コード

| コード | 意味 |
|---|---|
| `0` | 成功（回答を出力した） |
| `1` | エラー（APIキー未設定・次元不一致・AIサービスエラー等） |

### 典型的なワークフロー

```powershell
# 1. PDFをMarkdownに変換
uv run python convert.py report.pdf docs/

# 2. FAISSインデックスを構築
uv run python build_faiss.py --markdown-dir docs/report

# 3. 質問して回答を得る
uv run python rag_cli.py --index-folder faiss_index "質問テキスト"
```

---

---

## app.py — Streamlit GUI（ブラウザベースGUI）

PDF→Markdown変換・ベクトルDB構築・質問回答の3機能をブラウザ上のGUIで操作できる。

### 起動方法

```powershell
uv run streamlit run app.py
```

起動するとブラウザが自動で開き、GUIが表示される。

### 各タブの使い方

#### PDF→Markdown タブ

1. 「ファイルを選択」または「フォルダを選択」でPDFを指定する
2. 「サブフォルダを含める」でサブフォルダ内のPDFも対象にできる
3. 出力先フォルダを指定する
4. 「詳細設定」で変換オプションを調整できる（後述）
5. 「変換実行」ボタンをクリックすると進捗が表示される

**詳細設定オプション**:

| オプション | デフォルト | 説明 |
|---|---|---|
| 既存ファイルの扱い | 上書きする | 「スキップ」を選択すると出力.mdが既存の場合そのPDFをスキップする |
| 画像を抽出する | OFF | 図・数式をPNG画像として保存し、Markdownから相対パス参照する |
| 画像保存サブフォルダ名 | `images` | 画像の保存先サブフォルダ名（画像抽出有効時） |
| LLMで画像説明文を生成する | OFF | OpenAI APIで画像の説明文をMarkdownに追記する（画像抽出有効時） |
| OpenAIモデル名 | `gpt-5-mini` | 画像説明文生成に使用するモデル（説明文生成有効時） |
| 詳細ログを表示する | OFF | 変換処理の詳細ログをGUI上に表示する |

画像抽出を有効にした場合の出力構成:

```
output_dir/
└── document/
    ├── document.md
    └── images/
        ├── picture-1.png
        └── picture-2.png
```

#### DB構築タブ

1. 「ファイルを選択」または「フォルダを選択」でMarkdownファイルまたはフォルダを指定する
2. 「サブフォルダを含める」でサブフォルダ内のMarkdownも対象にできる
3. 「詳細設定」でオプションを調整できる
4. 「DB構築実行」ボタンをクリックするとFAISSインデックスが生成される

インデックスはMarkdownファイルごとに、そのファイルの親フォルダ内に作成される（例: `docs/chapter1/chapter1.md` → `docs/chapter1/faiss_index/`）。

**詳細設定オプション**:

| オプション | デフォルト | 説明 |
|---|---|---|
| 出力フォルダ名 | `faiss_index` | Markdownファイルの親フォルダ内に作成されるインデックスフォルダ名 |
| 既存DBの扱い | 上書き（再構築） | 「スキップ」を選択すると既存DBが存在する場合は構築処理をスキップする |
| 埋め込みモデル | `cl-nagoya/ruri-v3-30m` | HuggingFace埋め込みモデル名 |
| チャンクサイズ | `1000` | テキスト分割サイズ（文字数） |
| オーバーラップ | `200` | チャンクオーバーラップ（文字数） |

#### 質問回答タブ

1. 「フォルダを選択」で対象フォルダを指定する（インデックスフォルダを再帰検索する基点となるフォルダ）
2. 「詳細設定」でオプションを調整できる
3. 画面下部の入力欄に質問を入力してEnterで送信する
4. 回答はMarkdown形式でレンダリングされる（数式はLaTeX形式 `$...$` / `$$...$$` で出力）
5. 各回答の下にある「質問 N をコピー」（履歴）または「この質問・回答をコピー」（最新回答）を展開すると、その質問・回答をコピーできるテキストが表示される
6. 「会話履歴をクリア」ボタンで履歴をリセットできる
7. 「会話を保存」ボタンで会話履歴をMarkdownファイルとして保存できる（履歴がある場合のみ有効）

**詳細設定オプション**:

| オプション | デフォルト | 説明 |
|---|---|---|
| インデックスフォルダ名 | `faiss_index` | 対象フォルダ以下を再帰検索するFAISSインデックスフォルダ名 |
| 埋め込みモデル | `cl-nagoya/ruri-v3-30m` | FAISSロード時の埋め込みモデル（DB構築時と同一にすること） |
| Top-K | `5` | 検索で取得する上位件数 |
| LLMモデル | `gpt-5-mini` | 回答生成に使用するOpenAIモデル名 |
| 詳細ログを表示する | OFF | 処理の詳細ログをGUI上に表示する |

### 設定プロファイル

質問回答タブの「詳細設定」内に「設定プロファイル」セクションがある。現在の設定（全タブ共通）を名前付きでファイルに保存し、後で呼び出すことができる。プロファイルはプロジェクト直下の `config/` フォルダにJSONファイルとして保存される。

| 操作 | 説明 |
|---|---|
| プロファイルを選択して「読込」 | 保存済みプロファイルを選択して全設定を復元する |
| 名前を入力して「保存」 | 現在の全設定を指定名で `config/<名前>.json` に保存する |

### 設定の永続化

- 各タブの入力値はウィジェット変更時に自動で `settings.json` へ保存される
- アプリを再起動しても前回の設定が復元される
- 会話履歴はセッション内のみ保持され、再起動でリセットされる

### 環境変数

画像説明文生成（describe-images）または質問回答を使用する場合は `OPENAI_API_KEY` が必要。

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run streamlit run app.py
```

---

## テスト

```powershell
uv run pytest tests/ -v
```

## トラブルシューティング

| 問題 | 対処 |
|---|---|
| `エラー: ファイルが見つかりません` | ファイルパスを確認する |
| `エラー: OPENAI_API_KEY が設定されていません` | 環境変数 `OPENAI_API_KEY` を設定する |
| `エラー: --markdown-dir または --markdown-file を指定してください` | どちらか一方を指定する |
| `エラー: インデックス間でベクトル次元が一致しません` | 全インデックスを同一の埋め込みモデルで構築し直す |
| `エラー: AIサービスの呼び出しに失敗しました` | APIキーの有効性とネットワーク接続を確認する |
| 一部画像の説明文がない | `[WARNING]` メッセージを確認する（APIエラーによりスキップ） |
| `エラー: インデックス構築中にエラーが発生しました` | 埋め込みモデル名と接続環境を確認する |
