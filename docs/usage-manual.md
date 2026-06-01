# 詳細利用マニュアル

## 目次

1. [概要](#概要)
2. [前提条件・セットアップ](#前提条件セットアップ)
3. [CLIツール 詳細利用マニュアル](#cliツール-詳細利用マニュアル)
   - [convert.py — PDF → Markdown 変換](#convertpy--pdf--markdown-変換)
   - [build_faiss.py — FAISSインデックス構築](#build_faisspy--faissインデックス構築)
   - [rag_cli.py — RAQ 質問回答](#rag_clipy--rag-質問回答)
   - [CLIワークフロー全体例](#cliワークフロー全体例)
4. [GUI（app.py）詳細利用マニュアル](#guiapppy-詳細利用マニュアル)
   - [起動方法](#起動方法)
   - [PDF→Markdown タブ](#pdfmarkdown-タブ)
   - [DB構築 タブ](#db構築-タブ)
   - [質問回答 タブ](#質問回答-タブ)
   - [設定プロファイル管理](#設定プロファイル管理)
   - [GUIワークフロー全体例](#guiワークフロー全体例)
5. [CLIとGUIの動作比較](#cliとguiの動作比較)
6. [トラブルシューティング](#トラブルシューティング)

---

## 概要

このプロジェクトは、PDFドキュメントを検索可能な知識ベースに変換し、自然言語で質問回答を行う**RAG（Retrieval-Augmented Generation）システム**を提供する。

| ツール | 役割 |
|--------|------|
| `convert.py` | PDFをMarkdown形式に変換する（CLIのみ） |
| `build_faiss.py` | MarkdownをFAISSベクトルインデックスとして保存する（CLIのみ） |
| `rag_cli.py` | FAISSインデックスを検索してLLMが回答を生成する（CLIのみ） |
| `app.py` | 上記3機能をブラウザGUIで操作する |

---

## 前提条件・セットアップ

### 必要環境

- Python 3.13 以上
- [uv](https://github.com/astral-sh/uv)（パッケージマネージャー）
- `OPENAI_API_KEY` 環境変数（画像説明文生成・質問回答機能に必要）

### インストール

```powershell
uv sync
```

### APIキーの設定（セッション内）

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

---

## CLIツール 詳細利用マニュアル

### convert.py — PDF → Markdown 変換

#### 基本構文

```
uv run python convert.py INPUT_PDF OUTPUT_DIR [オプション]
```

| 引数 | 必須 | 説明 |
|------|------|------|
| `INPUT_PDF` | ✅ | 変換対象PDFファイルのパス |
| `OUTPUT_DIR` | ✅ | 出力先ディレクトリ（存在しない場合は自動作成） |

#### 出力構成

PDFと同名のサブフォルダが `OUTPUT_DIR` 内に作成され、その中にMarkdownが保存される。

```
OUTPUT_DIR/
└── <PDFのファイル名（拡張子なし）>/
    └── <PDFのファイル名>.md
```

例：`convert.py report.pdf output/` を実行した場合

```
output/
└── report/
    └── report.md
```

#### オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--extract-images` | フラグ | `False` | 図・数式をPNG画像として抽出し、Markdownから相対パス参照する |
| `--describe-images` | フラグ | `False` | OpenAI APIで画像の説明文を生成してMarkdownに追記する（`--extract-images` 必須） |
| `--model MODEL_NAME` | 文字列 | `gpt-5-mini` | 画像説明文生成に使用するOpenAIモデル名 |
| `--image-dir DIR` | 文字列 | `images` | 画像保存先サブディレクトリ名 |
| `--verbose` | フラグ | `False` | 処理の詳細ログをSTDERRに出力する |

#### 使用例

**基本変換**

```powershell
uv run python convert.py C:\docs\report.pdf C:\output\
```

標準出力に生成されたMarkdownのパスが表示される：

```
C:\output\report\report.md
```

**画像抽出あり**

```powershell
uv run python convert.py report.pdf output/ --extract-images
```

出力構成：

```
output/
└── report/
    ├── report.md
    └── images/
        ├── picture-1.png
        └── picture-2.png
```

**画像説明文生成あり**（OpenAI APIキー必要）

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run python convert.py report.pdf output/ --extract-images --describe-images
```

Markdownに画像説明文ブロックが挿入される：

```markdown
![picture-1.png](./images/picture-1.png)

> この図はシステムアーキテクチャを示しており、...
```

**詳細ログを表示しながら変換**

```powershell
uv run python convert.py report.pdf output/ --verbose 2>&1
```

#### 終了コードと対処

| コード | 意味 | 対処 |
|--------|------|------|
| `0` | 成功 | — |
| `1` | 入力ファイルエラー | ファイルパスを確認する |
| `2` | docling変換エラー | PDFが破損していないか確認する |
| `3` | APIキーエラー | `OPENAI_API_KEY` 環境変数を設定する |

---

### build_faiss.py — FAISSインデックス構築

`convert.py` が生成したMarkdownをベクトル化し、FAISSインデックスとして保存する。後続の `rag_cli.py` や `app.py` がインデックスを検索できるようになる。

#### 基本構文

```
uv run python build_faiss.py [オプション]
```

`--markdown-dir` または `--markdown-file` のどちらか（または両方）を必ず指定すること。

#### 出力構成

```
<input_dir または ファイルの親フォルダ>/
└── faiss_index/         ← --output-subdir で変更可
    ├── index.faiss
    └── index.pkl
```

#### オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--markdown-dir PATH` | 文字列 | — | Markdownフォルダ（サブフォルダも含め **常に再帰的** に全 `.md` ファイルを対象） |
| `--markdown-file FILE...` | 文字列（複数） | `[]` | 特定Markdownファイル（複数指定可） |
| `--embedding-model NAME` | 文字列 | `cl-nagoya/ruri-v3-30m` | HuggingFace埋め込みモデル名 |
| `--chunk-size N` | 整数 | `1000` | テキスト分割サイズ（文字数） |
| `--chunk-overlap N` | 整数 | `200` | チャンクオーバーラップ（文字数）。`chunk-size` より小さい値を指定すること |
| `--faiss-existing MODE` | `skip`/`overwrite` | `skip` | 既存インデックスの扱い |
| `--output-subdir NAME` | 文字列 | `faiss_index` | インデックス保存先サブフォルダ名 |

#### インデックスの保存先ルール

| 入力指定方法 | 保存先 |
|-------------|--------|
| `--markdown-dir ./docs` | `./docs/faiss_index/` |
| `--markdown-file` でファイルの親フォルダが1か所 | その親フォルダ内 `faiss_index/` |
| `--markdown-file` でファイルの親フォルダが複数 | カレントディレクトリ `./faiss_index/` |

**重要**: CLIでは、指定ディレクトリ内の全Markdownを1つのインデックスにまとめる（GUIとは異なる動作）。

#### 使用例

**フォルダ内全Markdownをインデックス化（再帰）**

```powershell
uv run python build_faiss.py --markdown-dir .\docs\report
```

**既存インデックスを上書き**

```powershell
uv run python build_faiss.py --markdown-dir .\docs\report --faiss-existing overwrite
```

**特定ファイルのみをインデックス化**

```powershell
uv run python build_faiss.py --markdown-file docs\report\report.md
```

**チャンクサイズをカスタマイズ**

```powershell
uv run python build_faiss.py --markdown-dir .\docs --chunk-size 500 --chunk-overlap 100
```

**別モデルで構築**（rag_cli.py 実行時も同じモデルを指定すること）

```powershell
uv run python build_faiss.py --markdown-dir .\docs --embedding-model intfloat/multilingual-e5-small
```

#### 終了コードと対処

| コード | 意味 | 対処 |
|--------|------|------|
| `0` | 成功 | — |
| `1` | エラー（引数・ファイル・処理エラー） | STDERRのエラーメッセージを確認する |

---

### rag_cli.py — RAG 質問回答

構築済みFAISSインデックスを検索し、OpenAI LLMが回答を生成する。

#### 基本構文

```
uv run python rag_cli.py [オプション] QUERY
```

`OPENAI_API_KEY` 環境変数が必須。

#### オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `QUERY` | 文字列 | — | 質問テキスト（**位置引数・必須**） |
| `--base-dir PATH` | 文字列 | カレントディレクトリ | インデックス探索を開始するルートフォルダ |
| `--index-folder NAME...` | 文字列（複数） | `faiss_index` | 探索するインデックスフォルダ名（複数指定可） |
| `--model NAME` | 文字列 | `gpt-5-mini` | 使用するAIモデル名 |
| `--top-k N` | 整数 | `5` | 検索で取得する上位チャンク件数（1以上） |
| `--embedding-model NAME` | 文字列 | `cl-nagoya/ruri-v3-30m` | 埋め込みモデル名（インデックス構築時と同じ値を指定すること） |
| `--verbose` | フラグ | `False` | Pythonライブラリの警告メッセージを表示する |

#### インデックス探索の仕組み

`--base-dir` に指定したフォルダ以下を再帰的に探索し、`--index-folder` で指定した名前のフォルダをすべて発見してメモリ上でマージする。例えば複数のPDFを別々のフォルダに変換・インデックス化した場合でも、1回のコマンドで横断検索できる。

```
base_dir/
├── project_a/
│   ├── report.md
│   └── faiss_index/          ← 発見される
│       ├── index.faiss
│       └── index.pkl
└── project_b/
    ├── manual.md
    └── faiss_index/          ← 発見される
        ├── index.faiss
        └── index.pkl
```

#### 出力形式

```
[回答]
RAG（Retrieval-Augmented Generation）とは、...

[参照情報]
source: C:\docs\report\report.md, chunk_id: 2
source: C:\docs\manual\manual.md, chunk_id: 0
```

#### インデックスが見つからない場合

インデックスが1件も見つからなかった場合、エラー終了せず以下の警告を表示してからインデックスなしでLLMに問い合わせる：

```
警告: 有効なインデックスが見つかりませんでした。インデックスなしでAIに問い合わせます。
注意: ドキュメントに基づかない回答が生成される可能性があります。
```

#### 使用例

**カレントディレクトリからインデックスを検索して質問**

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run python rag_cli.py "RAGとは何ですか？"
```

**特定フォルダを基点に検索**

```powershell
uv run python rag_cli.py --base-dir C:\projects\docs "設計の概要を教えてください"
```

**複数のインデックスフォルダ名を横断検索**

```powershell
uv run python rag_cli.py `
  --base-dir C:\projects `
  --index-folder faiss_index custom_index `
  "最新の仕様は何ですか？"
```

**上位取得件数とモデルをカスタマイズ**

```powershell
uv run python rag_cli.py --top-k 3 --model gpt-4o "詳細な技術仕様を教えてください"
```

#### 終了コードと対処

| コード | 意味 | 対処 |
|--------|------|------|
| `0` | 成功（回答を出力した） | — |
| `1` | APIキー未設定・次元不一致・AIサービスエラー等 | STDERRのエラーメッセージを確認する |

---

### CLIワークフロー全体例

```powershell
# 0. 環境設定
$env:OPENAI_API_KEY = "sk-..."

# 1. PDFをMarkdownに変換
uv run python convert.py C:\reports\annual.pdf C:\workspace\

# → C:\workspace\annual\annual.md が作成される

# 2. FAISSインデックスを構築
uv run python build_faiss.py --markdown-dir C:\workspace\annual

# → C:\workspace\annual\faiss_index\ が作成される

# 3. 質問して回答を得る
uv run python rag_cli.py --base-dir C:\workspace "今期の売上目標は何ですか？"
```

---

## GUI（app.py）詳細利用マニュアル

### 起動方法

```powershell
$env:OPENAI_API_KEY = "sk-..."
uv run streamlit run app.py
```

起動するとブラウザが自動で開き、GUIが表示される（デフォルト: `http://localhost:8501`）。

初回起動時に `settings.json` が自動生成される。アプリを終了しても設定は保持される。

---

### PDF→Markdown タブ

PDFファイルをMarkdown形式に変換するタブ。

#### 手順

1. **入力ファイル/フォルダを選択する**
   - 「ファイルを選択」: OSのファイル選択ダイアログでPDFを1件選択する
   - 「フォルダを選択」: PDFを含むフォルダを選択する
   - 「サブフォルダを含める」チェックボックスで再帰検索の有無を切り替える

2. **出力先フォルダを指定する**
   - 「フォルダを選択」でOSのフォルダ選択ダイアログを使うか、テキスト欄に直接入力する

3. **「詳細設定」を展開してオプションを設定する**（任意）

4. **「変換実行」ボタンをクリックする**
   - 変換中はステータスエリアにファイルごとの進捗が表示される
   - 進捗バーが完了するまで待機する

#### 詳細設定オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| 既存ファイルの扱い | 上書きする | 「スキップ」にすると既存の `.md` が存在するPDFをスキップする |
| 画像を抽出する | OFF | 図・数式をPNG画像として保存し、Markdownに参照を埋め込む |
| 画像保存サブフォルダ名 | `images` | 画像保存先のサブフォルダ名（画像抽出有効時） |
| LLMで画像説明文を生成する | OFF | OpenAI APIで各画像の説明文を生成してMarkdownに挿入する（`OPENAI_API_KEY` 必須・画像抽出有効時のみ） |
| OpenAIモデル名 | `gpt-5-mini` | 画像説明文生成に使用するモデル |
| 詳細ログを表示する | OFF | 処理の詳細ログをGUI上に表示する |

#### 出力構成

```
出力フォルダ/
└── <PDFのファイル名>/
    ├── <PDFのファイル名>.md
    └── images/              ← 画像抽出有効時のみ
        ├── picture-1.png
        └── picture-2.png
```

---

### DB構築 タブ

MarkdownファイルをFAISSベクトルインデックスとして保存するタブ。

**GUIのDB構築はファイル単位でインデックスを作成する**。各Markdownファイルに対して独立したインデックスフォルダが、そのMarkdownファイルの親フォルダ内に作成される。CLIの `build_faiss.py` とは動作が異なる（[CLIとGUIの動作比較](#cliとguiの動作比較)参照）。

#### 手順

1. **入力ファイル/フォルダを選択する**
   - 「ファイルを選択」: Markdownファイルを1件選択する
   - 「フォルダを選択」: Markdownを含むフォルダを選択する
   - 「サブフォルダを含める」チェックボックスで再帰検索の有無を切り替える

2. **「詳細設定」を展開してオプションを設定する**（任意）

3. **「DB構築実行」ボタンをクリックする**
   - ステータスエリアにファイルごとの進捗が表示される

#### 詳細設定オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| 出力フォルダ名 | `faiss_index` | 各Markdownファイルの親フォルダ内に作成されるインデックスフォルダ名 |
| 既存DBの扱い | 上書き（再構築） | 「スキップ」にすると既存DBがある場合は構築処理をスキップする |
| 埋め込みモデル | `cl-nagoya/ruri-v3-30m` | HuggingFace埋め込みモデル名（質問回答タブと同じ値を使用すること） |
| チャンクサイズ | `1000` | テキスト分割サイズ（文字数） |
| オーバーラップ | `200` | チャンクオーバーラップ（文字数） |

#### 出力構成例

```
docs/
├── chapter1/
│   ├── chapter1.md
│   └── faiss_index/           ← chapter1.md に対するインデックス
│       ├── index.faiss
│       └── index.pkl
└── chapter2/
    ├── chapter2.md
    └── faiss_index/           ← chapter2.md に対するインデックス
        ├── index.faiss
        └── index.pkl
```

---

### 質問回答 タブ

FAISSインデックスを検索してOpenAI LLMが回答を生成する会話形式のタブ。

#### 手順

1. **対象フォルダを指定する**
   - 「フォルダを選択」でインデックスを再帰検索する基点フォルダを指定する
   - 指定フォルダ以下のすべての `faiss_index` フォルダが自動で発見・マージされる

2. **「詳細設定」を展開してオプションを設定する**（任意）

3. **画面下部の入力欄に質問を入力してEnterを押す**

4. **回答を確認する**
   - 回答はMarkdown形式でレンダリングされる
   - 数式は LaTeX 形式（インライン: `$...$`、ブロック: `$$...$$`）でレンダリングされる
   - 回答の下に参照元ファイル名が表示される

#### 詳細設定オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| インデックスフォルダ名 | `faiss_index` | 再帰検索するインデックスフォルダ名 |
| 埋め込みモデル | `cl-nagoya/ruri-v3-30m` | DB構築時と同じモデルを指定すること |
| Top-K | `5` | 検索で取得する上位チャンク件数 |
| LLMモデル | `gpt-5-mini` | 回答生成に使用するOpenAIモデル名 |
| 詳細ログを表示する | OFF | 処理の詳細ログをGUI上に表示する |

#### 会話操作

| 操作 | 説明 |
|------|------|
| 質問入力 + Enter | 質問を送信して回答を得る |
| 「質問 N をコピー」展開 | 過去の質問・回答をコピー可能なテキストで表示する |
| 「この質問・回答をコピー」展開 | 最新の質問・回答をコピー可能なテキストで表示する |
| 「会話履歴をクリア」ボタン | 会話履歴をリセットする（セッション内のみ） |
| 「会話を保存」ボタン | 会話履歴をMarkdownファイルとしてOSのファイルダイアログで保存する |

#### FAISSキャッシュについて

対象フォルダや埋め込みモデルが同じ場合、インデックスはセッション内でキャッシュされる（再読み込みしない）。フォルダのタイムスタンプが変更された場合は自動的に再読み込みする。

---

### 設定プロファイル管理

全タブの設定を名前付きで保存・復元できる機能。質問回答タブの「詳細設定」 → 「設定プロファイル」セクションで操作する。

#### 保存

1. 「設定プロファイル」セクションを展開する
2. 右側のテキスト欄にプロファイル名を入力する（例: `project_a`）
3. 「保存」ボタンをクリックする
4. `config/project_a.json` に現在の全設定が保存される

#### 読み込み

1. 「設定プロファイル」セクションを展開する
2. ドロップダウンから保存済みプロファイルを選択する
3. 「読込」ボタンをクリックする
4. 全タブの設定が選択したプロファイルの内容に変更され、自動で `settings.json` に反映される

---

### GUIワークフロー全体例

```
1. アプリを起動する
   uv run streamlit run app.py

2. PDF→Markdown タブで変換する
   ① 「フォルダを選択」で C:\reports\ を選択
   ② 「サブフォルダを含める」をONにする
   ③ 出力先を C:\workspace\ に設定
   ④ 「変換実行」をクリック
   → C:\workspace\<各PDF名>\<PDF名>.md が生成される

3. DB構築 タブでインデックスを構築する
   ① 「フォルダを選択」で C:\workspace\ を選択
   ② 「サブフォルダを含める」をONにする
   ③ 「DB構築実行」をクリック
   → 各 .md ファイルの親フォルダに faiss_index/ が生成される

4. 質問回答 タブで質問する
   ① 「フォルダを選択」で C:\workspace\ を選択（インデックスを再帰検索する基点）
   ② 質問を入力して Enter
   → 全インデックスがマージされ、関連ドキュメントに基づいて回答が生成される
```

---

## CLIとGUIの動作比較

| 項目 | CLI（build_faiss.py） | GUI（app.py DB構築タブ） |
|------|----------------------|-------------------------|
| インデックス単位 | **フォルダ単位**（全Markdownを1つのインデックスにまとめる） | **ファイル単位**（各Markdownに個別のインデックス） |
| `--markdown-dir` の再帰性 | 常に再帰検索（オプションなし） | 「サブフォルダを含める」で切り替え可 |
| 既存インデックスのデフォルト動作 | スキップ（`skip`） | 上書き（`overwrite`） |
| 同時処理 | すべてのファイルをまとめてインデックス化 | ファイルごとに順次処理 |
| 埋め込みモデルのデフォルト | `cl-nagoya/ruri-v3-30m` | `cl-nagoya/ruri-v3-30m` |

---

## トラブルシューティング

| 問題 | 原因 | 対処 |
|------|------|------|
| `エラー: ファイルが見つかりません` | パスが誤っているかファイルが存在しない | ファイルパスを確認する |
| `エラー: OPENAI_API_KEY が設定されていません` | 環境変数未設定 | `$env:OPENAI_API_KEY = "sk-..."` を実行する |
| `エラー: --markdown-dir または --markdown-file を指定してください` | 必須オプションの指定漏れ | どちらか一方を必ず指定する |
| `エラー: インデックス間でベクトル次元が一致しません` | 異なる埋め込みモデルで構築されたインデックスをマージしようとした | 全インデックスを同一の埋め込みモデルで再構築する |
| `エラー: AIサービスの呼び出しに失敗しました` | APIキーが無効またはネットワーク障害 | APIキーの有効性とネットワーク接続を確認する |
| `エラー: インデックス構築中にエラーが発生しました` | 埋め込みモデルのダウンロード失敗など | ネットワーク接続と埋め込みモデル名を確認する |
| 一部画像の説明文が生成されない | OpenAI APIエラー（画像フォーマット非対応等） | STDERRの `[WARNING]` メッセージを確認する |
| GUIで「変換実行」が無効になっている | 別の処理が実行中 | 実行中の処理が完了するまで待機する |
| GUIで設定が反映されない | settings.json の書き込み権限がない | アプリの作業ディレクトリの書き込み権限を確認する |
| `ConnectionResetError` がSTDERRに表示される | Windows の asyncio/ProactorEventLoop の既知の動作 | 処理結果に影響しないため無視してよい |
