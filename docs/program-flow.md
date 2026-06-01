# プログラムフロー資料

## 目次

1. [システム全体のデータフロー](#システム全体のデータフロー)
2. [convert.py のフロー](#convertpy-のフロー)
3. [build_faiss.py のフロー](#build_faisspy-のフロー)
4. [rag_cli.py のフロー](#rag_clipy-のフロー)
5. [app.py（GUI）のフロー](#apppy-guiのフロー)
   - [アプリ起動・セッション初期化](#アプリ起動セッション初期化)
   - [PDF→Markdown 変換タブ](#pdfmarkdown-変換タブのフロー)
   - [DB構築タブ](#db構築タブのフロー)
   - [質問回答タブ](#質問回答タブのフロー)
6. [データ構造](#データ構造)
7. [CLIとGUIの処理比較](#cliとguiの処理比較)

---

## システム全体のデータフロー

```
PDF ファイル
    │
    ▼  [convert.py / app.py PDF→Markdown タブ]
    │   docling でテキスト・画像を抽出
    │   (任意) OpenAI API で画像説明文を生成
    │
Markdown ファイル (.md)
    │
    ▼  [build_faiss.py / app.py DB構築タブ]
    │   RecursiveCharacterTextSplitter でチャンク分割
    │   HuggingFaceEmbeddings でベクトル化
    │   FAISS.from_documents() でインデックス構築
    │   FAISS.save_local() でディスクに保存
    │
FAISSインデックス (index.faiss + index.pkl)
    │
    ▼  [rag_cli.py / app.py 質問回答タブ]
    │   FAISS.load_local() でインデックスをロード
    │   (複数インデックス) FAISS.merge_from() でメモリ上でマージ
    │   similarity_search() で関連チャンクを検索
    │   ChatOpenAI でコンテキストと質問から回答を生成
    │
回答テキスト + 参照情報
```

---

## convert.py のフロー

### コールスタック

```
main()
├── parse_args()            コマンドライン引数を解析する
├── validate_args()         引数のバリデーション（ファイル存在・APIキー確認）
└── convert_pdf()           変換処理のメイン関数
    ├── [extract_images=False の場合]
    │   └── DocumentConverter()      docling デフォルト変換器を初期化
    ├── [extract_images=True の場合]
    │   └── DocumentConverter(       docling 画像抽出モードで変換器を初期化
    │       PdfPipelineOptions(generate_picture_images=True))
    │
    ├── converter.convert(input_pdf)    PDFを変換してdocling Documentを生成
    ├── document.export_to_markdown()   docling DocumentをMarkdown文字列に変換
    │
    ├── [extract_images=True の場合]
    │   ├── document.iterate_items()    ドキュメント内の各要素を反復
    │   ├── [PictureItem の場合]
    │   │   ├── element.get_image()     画像データを取得
    │   │   ├── img.save()              PNG画像をファイルに保存
    │   │   └── [describe_images=True の場合]
    │   │       └── describe_image()    OpenAI API で画像説明文を生成
    │   │           └── base64エンコード → openai.chat.completions.create()
    │   └── build_markdown()            画像参照と説明文をMarkdownに埋め込む
    │
    └── save_markdown()                 Markdownをファイルに書き込む
        └── output_dir / pdf_stem / f"{pdf_stem}.md"
```

### 処理の流れ（フローチャート）

```
開始
  │
  ▼
parse_args() で引数解析
  │
  ▼
validate_args()
  ├─ PDFファイルが存在しない → 終了コード1で終了
  ├─ --describe-images なのに --extract-images なし → 終了コード1で終了
  └─ describe_images=True で OPENAI_API_KEY 未設定 → 終了コード3で終了
  │
  ▼
DocumentConverter を初期化（extract_images の有無でオプションが変わる）
  │
  ▼
converter.convert(input_pdf) でdocling変換
  ├─ 変換失敗 → 終了コード2で終了
  │
  ▼
document.export_to_markdown() でMarkdown文字列を生成
  │
  ▼
[extract_images=True?]
  ├─ No → そのままMarkdownを使用
  └─ Yes → 画像ディレクトリを作成
              iterate_items() で PictureItem を検索
              各 PictureItem を PNG として保存
              [describe_images=True?]
              ├─ No → 画像パスのみをMarkdownに挿入
              └─ Yes → OpenAI API で説明文を生成
                         生成失敗時は [WARNING] を出力してスキップ
                         画像パスと説明文をMarkdownに挿入
  │
  ▼
出力ディレクトリを作成し、Markdownファイルを書き込む
  │
  ▼
生成されたMarkdownのパスを標準出力に表示
  │
  ▼
終了コード0で終了
```

---

## build_faiss.py のフロー

### コールスタック

```
main()
├── parse_args()                 コマンドライン引数を解析する
├── validate_args()              引数のバリデーション
│   ├─ --markdown-dir または --markdown-file が未指定 → 終了コード1
│   ├─ パスが存在しない → 終了コード1
│   └─ chunk_size <= chunk_overlap → 終了コード1
│
├── collect_markdown_files()     対象Markdownファイルを収集する
│   ├─ [--markdown-dir の場合] dir_path.glob("**/*.md")  ← 常に再帰
│   ├─ [--markdown-file の場合] 指定されたファイルパスをそのまま使用
│   └─ 重複排除（resolveして seen dict で管理）
│
├── _determine_output_dir()      インデックス保存先を決定する
│   ├─ [--markdown-dir の場合] Path(markdown_dir) / output_subdir
│   ├─ [全ファイルが同一親の場合] 親フォルダ / output_subdir
│   └─ [異なる親が混在の場合] Path.cwd() / output_subdir
│
├── [既存インデックスの確認]
│   ├─ [faiss_existing="skip"] → 終了コード0で終了（スキップ）
│   └─ [faiss_existing="overwrite"] → shutil.rmtree() で削除
│
├── ファイルを順に読み込んで Document オブジェクトを生成
│
├── split_documents()            テキストをチャンクに分割する
│   ├─ RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
│   └─ 各チャンクに chunk_id メタデータを付与（ファイル内の連番）
│
└── build_and_save_index()       インデックスを構築してディスクに保存する
    ├─ HuggingFaceEmbeddings(model_name)   埋め込みモデルをロード
    ├─ FAISS.from_documents(chunks, embeddings)  インデックスを構築
    └─ vectorstore.save_local(output_dir)  index.faiss + index.pkl を保存
```

### 処理の流れ（フローチャート）

```
開始
  │
  ▼
parse_args() → validate_args()
  ├─ バリデーション失敗 → 終了コード1
  │
  ▼
collect_markdown_files()
  │
  ▼
対象ファイルが0件?
  ├─ Yes → 終了コード1
  │
  ▼
_determine_output_dir() でインデックス保存先を決定
  │
  ▼
出力先が既に存在する?
  ├─ No → 続行
  ├─ Yes・faiss_existing="skip" → 終了コード0
  └─ Yes・faiss_existing="overwrite" → shutil.rmtree() で削除して続行
  │
  ▼
ファイルを順に読み込んで Document リストを生成
（UTF-8デコード失敗のファイルはスキップ）
  │
  ▼
Document が0件?
  ├─ Yes → 終了コード1
  │
  ▼
split_documents() でチャンク分割
  │
  ▼
HuggingFaceEmbeddings でモデルをロード
  │
  ▼
FAISS.from_documents() でインデックスを構築
  ├─ 失敗 → 作成途中のディレクトリを削除して終了コード1
  │
  ▼
save_local() でディスクに保存
  │
  ▼
保存先パスを標準出力に表示 → 終了コード0
```

---

## rag_cli.py のフロー

### コールスタック

```
main()
├── parse_args()               コマンドライン引数を解析する
│
├── [--verbose 未指定の場合]
│   └── warnings.filterwarnings("ignore")  ライブラリ警告を抑制（import前に設定）
│
├── [引数バリデーション]
│   ├─ query が空文字 → 終了コード1
│   ├─ top_k < 1 → 終了コード1
│   └─ validate_api_key()
│       └─ OPENAI_API_KEY 未設定 → 終了コード1
│
├── base_path の決定
│   ├─ [--base-dir 指定あり] Path(base_dir)
│   │   └─ ディレクトリが存在しない → 終了コード1
│   └─ [--base-dir 未指定] Path.cwd()
│
├── find_index_dirs()          インデックスフォルダを再帰探索する
│   └─ base_path.rglob(folder_name) で各フォルダ名を探索
│       有効条件: index.faiss と index.pkl が両方存在すること
│       見つからない場合: 警告を表示してそのフォルダ名をスキップ
│
├── [インデックスが0件の場合]
│   └─ _run_llm_query(query, "", model)  コンテキストなしでLLMに問い合わせ
│       print_result(answer, [], no_index_warning=True)
│       → 終了
│
├── _load_embeddings()         埋め込みモデルをロードする
│   ├─ local_files_only=True でキャッシュから優先ロード
│   └─ 失敗した場合は HF Hub からダウンロード
│
├── load_and_merge_indices()   全インデックスをロードしてマージする
│   ├─ FAISS.load_local() で各インデックスをロード
│   └─ FAISS.merge_from() でメモリ上にマージ
│       次元不一致の場合 → 終了コード1
│
├── run_query()                RAGクエリを実行する
│   ├─ merged_store.as_retriever(k=top_k)
│   ├─ retriever.invoke(query)  ← 類似度検索でTop-K チャンクを取得
│   ├─ format_context()         チャンクをプロンプト用テキストに整形
│   └─ _run_llm_query()         LLMに問い合わせる
│       ├─ SystemMessage + HumanMessage（コンテキスト + 質問）を構築
│       └─ ChatOpenAI(model).invoke(messages)
│           失敗 → 終了コード1
│
└── print_result()             回答と参照情報を標準出力に表示する
    ├─ [回答] セクション: answer テキスト
    └─ [参照情報] セクション: source + chunk_id を列挙
```

### 処理の流れ（フローチャート）

```
開始
  │
  ▼
parse_args()
  │
  ▼
バリデーション（query, top_k, OPENAI_API_KEY）
  ├─ 失敗 → 終了コード1
  │
  ▼
base_path を決定（--base-dir or cwd）
  │
  ▼
find_index_dirs() で有効なインデックスを再帰探索
  │
  ▼
インデックスが0件?
  ├─ Yes → _run_llm_query(query, context="") でコンテキストなし回答
  │          print_result(..., no_index_warning=True)
  │          → 終了コード0
  │
  ▼
_load_embeddings() で埋め込みモデルをロード
  │
  ▼
load_and_merge_indices()
  ├─ 次元不一致 → 終了コード1
  │
  ▼
run_query()
  ├─ similarity_search() で Top-K チャンクを取得
  └─ ChatOpenAI で回答を生成
      └─ APIエラー → 終了コード1
  │
  ▼
print_result() で標準出力に表示
  │
  ▼
終了コード0
```

---

## app.py（GUI）のフロー

### アプリ起動・セッション初期化

```
streamlit run app.py
  │
  ▼
モジュールレベルの処理
  ├─ os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  ← 起動時に必ず設定
  └─ 定数の定義（SETTINGS_FILE, CONFIG_DIR）
  │
  ▼
main() が呼ばれる（Streamlitが毎回のレンダリングで呼び出す）
  │
  ▼
init_session_state()          初回のみセッションステートを初期化
  ├─ "settings": load_settings()   ← settings.json から設定を読み込む
  │               ファイルが存在しない・不正JSON の場合はデフォルト設定を使用
  ├─ "running": False
  ├─ "chat_history": []
  ├─ "faiss_store": None
  ├─ "faiss_cache_key": ""
  └─ overwrite確認フラグ各種
  │
  ▼
ペンディングパスの適用
  └─ _pending_{key} が存在すれば対応するウィジェットのセッションキーに書き込む
      （tkinterダイアログで選択したパスをウィジェットに反映するため）
  │
  ▼
ペンディング設定読み込みの適用
  └─ "pending_config_load" が存在すれば設定を全ウィジェットに反映して st.rerun()
  │
  ▼
st.set_page_config() でページ設定
st.title() でタイトル表示
  │
  ▼
st.tabs(["PDF→Markdown", "DB構築", "質問回答"]) でタブを生成
  ├─ render_convert_tab()   PDF→Markdown タブを描画
  ├─ render_build_tab()     DB構築タブを描画
  └─ render_chat_tab()      質問回答タブを描画
```

### PDF→Markdown 変換タブのフロー

```
render_convert_tab()
  │
  ▼
入力パス・出力パスのウィジェットを描画
  └─ ファイル/フォルダ選択ボタン → pick_file() / pick_folder()（tkinter）
       選択後 _queue_path() でペンディングに登録 → st.rerun() で反映
  │
  ▼
「詳細設定」エクスパンダーを描画
  └─ overwrite_mode, extract_images, describe_images, model_name, image_dir, verbose
     ウィジェット変更時に save_settings() を呼ぶ
  │
  ▼
「変換実行」ボタン
  │
  ▼
validate_convert_inputs() バリデーション
  ├─ 失敗 → st.error() を表示して終了
  │
  ▼
st.session_state["running"] = True  ← 同時実行防止
  │
  ▼
with st.status("PDF変換中...", expanded=True) as status:
  │
  ▼
run_conversion(settings, status_container, progress_bar)
  │
  ├─ collect_pdf_files() で対象PDFを収集（recursive フラグに応じて）
  ├─ ファイルが0件 → ProcessResult(success=False, ...)
  │
  └─ DocumentConverter を初期化（extract_images の有無でオプションが変わる）
       各PDFに対して:
       ├─ [overwrite_mode="skip" かつ出力.md が存在する] → スキップ
       ├─ converter.convert(pdf) でdocling変換
       ├─ export_to_markdown() でMarkdown生成
       ├─ [extract_images=True]
       │   ├─ PictureItem を順に抽出してPNG保存
       │   └─ [describe_images=True かつ OPENAI_API_KEY あり]
       │       └─ _describe_image_with_openai() で説明文生成
       ├─ out_file.parent.mkdir() / out_file.write_text()
       └─ progress_bar.progress() で進捗更新
  │
  ▼
ProcessResult に応じて status.update()
  ├─ success=True → state="complete"
  └─ success=False → state="error"
  │
  ▼
st.session_state["running"] = False
```

### DB構築タブのフロー

```
render_build_tab()
  │
  ▼
入力パスのウィジェットを描画
  └─ ファイル/フォルダ選択ボタン → pick_file() / pick_folder()（tkinter）
  │
  ▼
「詳細設定」エクスパンダーを描画
  └─ output_dir, faiss_existing, embedding_model, chunk_size, chunk_overlap
  │
  ▼
「DB構築実行」ボタン
  │
  ▼
validate_build_inputs() バリデーション
  ├─ 失敗 → st.error() を表示して終了
  │
  ▼
st.session_state["running"] = True
  │
  ▼
with st.status("DB構築中...", expanded=True) as status:
  │
  ▼
run_build(settings, status_container, progress_bar)
  │
  ├─ collect_md_files() で対象Markdownを収集（recursive フラグに応じて）
  ├─ ファイルが0件 → ProcessResult(success=False, ...)
  │
  ├─ HuggingFaceEmbeddings(embedding_model) でモデルをロード
  ├─ RecursiveCharacterTextSplitter を初期化
  │
  └─ 各Markdownファイルに対して（ファイル単位で独立したインデックス）:
       ├─ output_dir = md_file.parent / output_dir_name
       ├─ [faiss_existing="skip" かつ output_dir が存在する] → スキップ
       ├─ md_file.read_text() でMarkdownを読み込む
       ├─ splitter.split_documents() でチャンク分割
       ├─ [output_dir が存在する場合] shutil.rmtree() で削除
       ├─ FAISS.from_documents(chunks, embeddings) でインデックス構築
       ├─ store.save_local(output_dir) で保存
       └─ progress_bar.progress() で進捗更新
  │
  ▼
ProcessResult に応じて status.update()
  │
  ▼
st.session_state["running"] = False
```

### 質問回答タブのフロー

```
render_chat_tab()
  │
  ▼
「設定プロファイル」エクスパンダーを描画
  └─ list_config_files() でプロファイル一覧を取得（config/*.json）
     「読込」ボタン → load_config() → _apply_settings_to_widgets() → st.rerun()
     「保存」ボタン → save_config() で config/<name>.json に保存
  │
  ▼
対象フォルダのウィジェットを描画
  └─ フォルダ選択ボタン → pick_folder()（tkinter）
  │
  ▼
「詳細設定」エクスパンダーを描画
  └─ index_folder, embedding_model, top_k, llm_model, verbose
  │
  ▼
対象フォルダの情報を st.caption() で表示
  │
  ▼
会話履歴（chat_history）を表示
  └─ Q&Aペアごとにユーザーメッセージと回答を描画
     各ペアに「コピー用テキスト」エクスパンダーを追加
  │
  ▼
「会話履歴をクリア」「会話を保存」ボタンを描画
  └─ 「会話を保存」→ pick_save_file()（tkinter）でMarkdownとして保存
  │
  ▼
st.chat_input() で質問入力を待機
  │
  ▼（質問が入力された場合）
validate_chat_inputs() バリデーション
  ├─ 失敗 → st.error() を表示して終了
  │
  ▼
with st.spinner("回答を生成中..."):
  │
  ▼
load_faiss_if_needed(base_dir, index_folder, embedding_model)
  ├─ get_faiss_cache_key() でキャッシュキーを生成
  │   キー = "base_dir|index_folder|model|mtime"
  ├─ キャッシュキーが一致する場合 → キャッシュされた faiss_store を返す
  └─ キャッシュミスの場合:
       find_index_dirs() で有効なインデックスフォルダを再帰探索
       load_and_merge_indices() で全インデックスをロード・マージ
       セッションステートにキャッシュを保存
  │
  ▼
run_query(query, faiss_store, chat_history, settings)
  ├─ faiss_store.similarity_search(query, k=top_k) で関連チャンクを取得
  ├─ メッセージリストを構築:
  │   SystemMessage（コンテキスト付き）
  │   + chat_history の全メッセージ（HumanMessage / AIMessage）
  │   + 今回の HumanMessage(query)
  └─ ChatOpenAI(model).invoke(messages) で回答を生成
  │
  ▼
chat_history に質問・回答・ソース情報を追加
  │
  ▼
st.rerun() で会話履歴を再描画
```

---

## データ構造

### settings.json（設定永続化ファイル）

```json
{
  "convert": {
    "input_path": "",
    "output_dir": "",
    "recursive": false,
    "overwrite_mode": "overwrite",
    "extract_images": false,
    "describe_images": false,
    "model_name": "gpt-5-mini",
    "image_dir": "images",
    "verbose": false
  },
  "build": {
    "input_path": "",
    "output_dir": "faiss_index",
    "recursive": false,
    "embedding_model": "cl-nagoya/ruri-v3-30m",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "faiss_existing": "overwrite"
  },
  "chat": {
    "index_folder": "faiss_index",
    "embedding_model": "cl-nagoya/ruri-v3-30m",
    "top_k": 5,
    "llm_model": "gpt-5-mini",
    "base_dir": "",
    "verbose": false
  }
}
```

### FAISSインデックス（ディスク上）

```
faiss_index/
├── index.faiss    ベクトルデータ本体（FAISS バイナリ形式）
└── index.pkl      メタデータ（ソースパス・chunk_id など）
```

各チャンクのメタデータ：

```python
{
    "source": "C:/path/to/document.md",  # 元ファイルの絶対パス
    "chunk_id": 0                         # ファイル内での連番（0始まり）
}
```

### セッションステート（app.py、セッション内のみ）

| キー | 型 | 説明 |
|------|-----|------|
| `settings` | `dict` | 全設定（settings.json と同期） |
| `running` | `bool` | 処理実行中フラグ（同時実行防止） |
| `chat_history` | `list[dict]` | 会話履歴（`{"role": "user"/"assistant", "content": str}`） |
| `faiss_store` | `FAISS \| None` | キャッシュされたFAISSストア |
| `faiss_cache_key` | `str` | キャッシュ検証用キー（パス+モデル+mtime） |
| `_pending_{key}` | `str` | tkinterダイアログ選択後のパス（次回描画時に適用） |
| `pending_config_load` | `dict \| None` | プロファイル読み込み待ちの設定 |

---

## CLIとGUIの処理比較

### DB構築処理の違い

```
build_faiss.py（CLI）               app.py（GUI DB構築タブ）
─────────────────────               ────────────────────────
複数ファイルを収集                     複数ファイルを収集
        │                                   │
        ▼                                   ▼
全チャンクを結合                         各ファイルを順に処理
        │                                   │
        ▼                              ┌────┴────┐
FAISS.from_documents()             チャンク分割
（1つのインデックス）               FAISS.from_documents()
        │                           save_local(md.parent/output_dir)
        ▼                           （ファイルごとに個別インデックス）
save_local(input_dir/faiss_index)  └────────────┘
（まとめて1つ保存）
```

### インデックス探索と検索処理の類似性

rag_cli.py と app.py 質問回答タブは、FAISSインデックスの探索・マージ・検索の基本ロジックは同じだが、app.py ではキャッシュ機構が追加されている。

```
rag_cli.py                          app.py 質問回答タブ
──────────                          ──────────────────
毎回 find_index_dirs() で探索       キャッシュキー比較
毎回インデックスをロード              ├─ キャッシュヒット → 保存済みを使用
毎回 merge_from() でマージ          └─ キャッシュミス → ロード・マージ・保存
```
